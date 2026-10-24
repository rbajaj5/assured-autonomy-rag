"""RAG-assisted rapid classifier for code/repo excerpts.

This creates quiz-style examples:

    code/README excerpt + retrieved module passages -> compact assurance label

The labels can be used immediately as a router dataset and later to train a
small local classifier. The goal is not final synthesis; it is fast first-pass
classification of assurance role, module fit, CPS escalation point, missing
evidence, and overclaim warning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from .config import AA_MODULES
from .indexing import RagIndex
from .llm import get_llm


DEFAULT_REPO_ROOT = Path("external_repos")
DEFAULT_OUT_DIR = Path("rapid_classifier_runs")
CODE_EXTS = {
    ".py", ".md", ".rs", ".js", ".ts", ".tsx", ".jsx", ".yaml", ".yml",
    ".toml", ".json", ".cpp", ".hpp", ".c", ".h", ".java", ".m", ".sh",
    ".clj", ".cljc", ".cljs", ".scm", ".ss", ".rkt", ".edn",
}
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    "target", ".mypy_cache", ".pytest_cache", "checkpoints", "weights", "models",
}


def _json_from_text(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def _repo_name(path: Path) -> str:
    return path.name


def _iter_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix.lower() not in CODE_EXTS:
            continue
        try:
            if path.stat().st_size > 240_000:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _sample_excerpt(text: str, *, max_chars: int, rng: random.Random) -> str:
    text = re.sub(r"\n{4,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    windows = []
    for match in re.finditer(r"(?m)^(def |class |function |export |pub fn |fn |# |## |### )", text):
        start = max(match.start() - 200, 0)
        windows.append(start)
    if not windows:
        windows = list(range(0, max(len(text) - max_chars, 1), max_chars))
    start = rng.choice(windows)
    return text[start:start + max_chars].strip()


def sample_repo_excerpts(
    repo_roots: list[Path],
    *,
    samples_per_repo: int,
    max_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for repo in repo_roots:
        files = _iter_files(repo)
        if not files:
            continue
        priority = sorted(
            files,
            key=lambda p: (
                0 if p.name.lower() in {"readme.md", "readme.rst"} else 1,
                0 if p.suffix.lower() in {".py", ".rs", ".cpp", ".md"} else 1,
                len(str(p)),
            ),
        )
        chosen = priority[: min(8, len(priority))]
        if len(files) > len(chosen):
            chosen += rng.sample(files, min(max(samples_per_repo * 3, 6), len(files)))
        seen: set[Path] = set()
        for path in chosen:
            if path in seen:
                continue
            seen.add(path)
            text = _read_text(path)
            if len(text.strip()) < 220:
                continue
            rel = str(path.relative_to(repo))
            examples.append({
                "repo": _repo_name(repo),
                "repo_path": str(repo),
                "file": rel,
                "excerpt": _sample_excerpt(text, max_chars=max_chars, rng=rng),
            })
            if sum(1 for ex in examples if ex["repo"] == _repo_name(repo)) >= samples_per_repo:
                break
    return examples


def sample_repo_sequences(
    repo_roots: list[Path],
    *,
    sequences_per_repo: int,
    snippets_per_sequence: int,
    max_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sequences: list[dict[str, Any]] = []
    for repo in repo_roots:
        files = _iter_files(repo)
        if not files:
            continue
        readmes = [p for p in files if p.name.lower().startswith("readme")]
        code_files = [p for p in files if p.suffix.lower() not in {".md", ".json", ".yaml", ".yml", ".toml"}]
        config_files = [p for p in files if p.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}]
        ordered_pool = sorted(readmes, key=lambda p: len(str(p)))
        ordered_pool += sorted(code_files, key=lambda p: (len(p.parts), str(p)))[:80]
        ordered_pool += sorted(config_files, key=lambda p: (len(p.parts), str(p)))[:30]
        ordered_pool = list(dict.fromkeys(ordered_pool))
        if not ordered_pool:
            continue
        for seq_idx in range(sequences_per_repo):
            chosen: list[Path] = []
            if readmes:
                chosen.append(readmes[0])
            remaining = [p for p in ordered_pool if p not in chosen]
            if remaining:
                chosen.extend(rng.sample(remaining, min(snippets_per_sequence - len(chosen), len(remaining))))
            snippets = []
            for pos, path in enumerate(chosen[:snippets_per_sequence], 1):
                text = _read_text(path)
                if len(text.strip()) < 160:
                    continue
                snippets.append({
                    "position": pos,
                    "file": str(path.relative_to(repo)),
                    "excerpt": _sample_excerpt(text, max_chars=max_chars, rng=rng),
                })
            if snippets:
                sequences.append({
                    "repo": _repo_name(repo),
                    "repo_path": str(repo),
                    "sequence_id": f"{_repo_name(repo)}::{seq_idx:03d}",
                    "snippets": snippets,
                })
    return sequences


def retrieve_module_excerpts(index: RagIndex, query: str, *, k: int) -> list[dict[str, Any]]:
    hits = index.query(query, k=k, min_score=0.0)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        ch = hit.chunk
        key = (ch.module_id, ch.filename)
        if key in seen and len(rows) >= max(4, k // 2):
            continue
        seen.add(key)
        rows.append({
            "module_id": ch.module_id,
            "module_name": ch.module_name,
            "filename": ch.filename,
            "citation": ch.citation(),
            "score": round(hit.score, 4),
            "text": ch.text[:900].strip(),
        })
    return rows[:k]


def _label_status(label: dict[str, Any]) -> str:
    return "complete" if label.get("primary_module") else "failed"


def classify_excerpt(
    *,
    llm: Any,
    example: dict[str, Any],
    module_excerpts: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    module_catalog = "\n".join(f"- {mid}: {name}" for mid, name in AA_MODULES.items())
    system = (
        "You are a rapid Assured Autonomy classifier for a Hopkins cybersecurity-track student. "
        "Classify the code/repo excerpt using the retrieved course passages as RAG context. "
        "This is a quiz, not final synthesis. Return compact valid JSON only."
    )
    user = (
        "Canonical modules:\n"
        f"{module_catalog}\n\n"
        "Repo excerpt:\n"
        f"repo: {example['repo']}\nfile: {example['file']}\n"
        f"```\n{example['excerpt']}\n```\n\n"
        "Retrieved course passages:\n"
        f"{json.dumps(module_excerpts, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with exactly these fields:\n"
        "{\n"
        '  "primary_module": "M12",\n'
        '  "secondary_modules": ["M06"],\n'
        '  "assurance_role": "runtime monitor | scenario generator | safe controller | perception baseline | adversarial stressor | data/shift detector | governance/audit | other",\n'
        '  "hidden_witness": "...",\n'
        '  "cyberphysical_escalation_point": "...",\n'
        '  "missing_evidence": ["..."],\n'
        '  "overclaim_warning": "...",\n'
        '  "risk_portfolio": {\n'
        '    "risky_assets": ["..."],\n'
        '    "risk_factors": ["..."],\n'
        '    "controls_or_hedges": ["..."],\n'
        '    "evidence_assets": ["..."],\n'
        '    "residual_risk": ["..."],\n'
        '    "portfolio_role": "what this excerpt contributes to managing risk"\n'
        "  },\n"
        '  "confidence": "low | medium | high",\n'
        '  "rag_support_status": "supported | inferred_missing_primary_passage | weak",\n'
        '  "rationale": "2-4 concise sentences tied to the excerpt and RAG passages",\n'
        '  "supporting_passages": [{"module": "M12", "citation": "[M12 / file.pdf, p. 1]", "supports": "..."}],\n'
        '  "hard_negative_modules": [{"module": "M05", "why_not_primary": "..."}]\n'
        "}\n"
        "Rules:\n"
        "- supporting_passages must use only citations that appear verbatim in Retrieved course passages.\n"
        "- If the best primary_module is not represented in the retrieved passages, keep it only if the excerpt strongly justifies it and set rag_support_status to inferred_missing_primary_passage.\n"
        "- Do not invent files, papers, or citation handles.\n"
    )
    raw = _complete_with_retry(
        llm,
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens,
        retries=getattr(llm, "_rapid_retries", 0),
        sleep_seconds=getattr(llm, "_rapid_sleep_seconds", 10.0),
    )
    label = _json_from_text(raw)
    label.update({
        "raw_model_output": raw,
        "status": _label_status(label),
    })
    return label


def _complete_with_retry(
    llm: Any,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return llm.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()
            retryable = any(term in message for term in ["429", "rate", "timeout", "temporarily", "overloaded"])
            if attempt >= retries or not retryable:
                raise
            wait = sleep_seconds * (attempt + 1)
            _safe_print(f"  ! LLM call failed on attempt {attempt + 1}: {str(exc)[:220]}; sleeping {wait:.1f}s")
            time.sleep(wait)
    raise last_exc or RuntimeError("LLM call failed")


def classify_sequence(
    *,
    llm: Any,
    sequence: dict[str, Any],
    module_excerpts: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    module_catalog = "\n".join(f"- {mid}: {name}" for mid, name in AA_MODULES.items())
    snippet_text = "\n\n".join(
        f"[{s['position']}] file: {s['file']}\n```\n{s['excerpt']}\n```"
        for s in sequence["snippets"]
    )
    system = (
        "You are a rapid Assured Autonomy sequence classifier. Treat ordered code snippets "
        "as a trace through a repo, like a short program with positional evidence. Classify "
        "the repo-level assurance thought induced by the sequence. Return compact valid JSON only."
    )
    user = (
        "Canonical modules:\n"
        f"{module_catalog}\n\n"
        f"Repo: {sequence['repo']}\nSequence id: {sequence['sequence_id']}\n\n"
        "Ordered snippets:\n"
        f"{snippet_text}\n\n"
        "Retrieved course passages:\n"
        f"{json.dumps(module_excerpts, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with exactly these fields:\n"
        "{\n"
        '  "primary_module": "M12",\n'
        '  "secondary_modules": ["M07"],\n'
        '  "repo_assurance_role": "scenario/evaluation layer | runtime monitor layer | safe-control layer | perception layer | adversarial stress layer | governance/audit layer | other",\n'
        '  "sequence_thought": "one compact statement of what the ordered snippets imply",\n'
        '  "positional_evidence": [{"position": 1, "file": "...", "supports": "..."}],\n'
        '  "hidden_witnesses": ["..."],\n'
        '  "cyberphysical_escalation_points": ["..."],\n'
        '  "missing_evidence": ["..."],\n'
        '  "overclaim_warnings": ["..."],\n'
        '  "trust_transfer_questions": ["..."],\n'
        '  "risk_portfolio": {\n'
        '    "risky_assets": ["vehicle trajectory safety", "sensor integrity", "fleet coordination"],\n'
        '    "risk_factors": ["dataset shift", "runtime fault", "adversarial perception"],\n'
        '    "exposure_vector": {"dataset_shift": "high", "runtime_fault": "medium"},\n'
        '    "controls_or_hedges": ["runtime monitor", "safe controller", "scenario diversification"],\n'
        '    "evidence_assets": ["logs", "scenario results", "monitor verdicts"],\n'
        '    "residual_risk": ["..."],\n'
        '    "portfolio_role": "what this sequence contributes to the assurance risk portfolio"\n'
        "  },\n"
        '  "rag_support_status": "supported | inferred_missing_primary_passage | weak",\n'
        '  "supporting_passages": [{"module": "M12", "citation": "[M12 / file.pdf, p. 1]", "supports": "..."}],\n'
        '  "hard_negative_modules": [{"module": "M05", "why_not_primary": "..."}],\n'
        '  "confidence": "low | medium | high"\n'
        "}\n"
        "Rules:\n"
        "- supporting_passages must use only citations that appear verbatim in Retrieved course passages.\n"
        "- Use positional_evidence to show why sequence order matters.\n"
        "- If the primary module is inferred from code but not directly retrieved, set rag_support_status to inferred_missing_primary_passage.\n"
        "- Do not invent citations.\n"
    )
    raw = _complete_with_retry(
        llm,
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens,
        retries=getattr(llm, "_rapid_retries", 0),
        sleep_seconds=getattr(llm, "_rapid_sleep_seconds", 10.0),
    )
    label = _json_from_text(raw)
    label.update({
        "raw_model_output": raw,
        "status": _label_status(label),
    })
    return label


def generate_quiz_for_excerpt(
    *,
    llm: Any,
    example: dict[str, Any],
    module_excerpts: list[dict[str, Any]],
    classifier_label: dict[str, Any] | None = None,
    max_tokens: int,
) -> dict[str, Any]:
    module_catalog = "\n".join(f"- {mid}: {name}" for mid, name in AA_MODULES.items())
    system = (
        "You write multiple-choice quiz items for a Hopkins Assured Autonomy compiler. "
        "The quiz teaches a model to distinguish nearby assurance concepts from code. "
        "Return compact valid JSON only."
    )
    user = (
        "Canonical modules:\n"
        f"{module_catalog}\n\n"
        "Code excerpt:\n"
        f"repo: {example['repo']}\nfile: {example['file']}\n"
        f"```\n{example['excerpt']}\n```\n\n"
        "Retrieved course passages. Every answer choice must cite only one of these citations verbatim:\n"
        f"{json.dumps(module_excerpts, ensure_ascii=False, indent=2)}\n\n"
        "Prior classifier label. The correct answer should align with this label unless it is weak or contradicted by retrieved evidence:\n"
        f"{json.dumps(classifier_label or {}, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with exactly these fields:\n"
        "{\n"
        '  "question_type": "module_classification | risk_control | overclaim_boundary | trust_transfer | hidden_witness",\n'
        '  "question": "...",\n'
        '  "choices": [\n'
        '    {"label": "A", "answer": "...", "module": "M12", "citation": "[M12 / file.pdf, p. 1]", "is_correct": true, "why": "..."},\n'
        '    {"label": "B", "answer": "...", "module": "M05", "citation": "[M05 / file.pdf, p. 2]", "is_correct": false, "why": "..."}\n'
        "  ],\n"
        '  "correct_label": "A",\n'
        '  "explanation": "...",\n'
        '  "hard_negative_logic": "why the wrong choices are plausible but wrong",\n'
        '  "risk_portfolio_lesson": "...",\n'
        '  "confidence": "low | medium | high",\n'
        '  "rag_support_status": "supported | weak"\n'
        "}\n"
        "Rules:\n"
        "- Create exactly four choices A-D.\n"
        "- Exactly one choice must be correct.\n"
        "- Each choice must cite a retrieved citation verbatim. Do not invent citations.\n"
        "- Distractors should be plausible adjacent modules, not silly answers.\n"
        "- Prefer questions that expose an overclaim boundary or risk-management distinction.\n"
        "- If rag_support_status in the classifier label is inferred_missing_primary_passage, it is acceptable to ask an overclaim-boundary question where the correct answer says the primary claim is inferred rather than directly retrieved.\n"
    )
    raw = _complete_with_retry(
        llm,
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens,
        retries=getattr(llm, "_rapid_retries", 0),
        sleep_seconds=getattr(llm, "_rapid_sleep_seconds", 10.0),
    )
    quiz = _json_from_text(raw)
    quiz.update({"raw_model_output": raw, "status": "complete" if quiz.get("question") else "failed"})
    return quiz


def generate_quiz_for_sequence(
    *,
    llm: Any,
    sequence: dict[str, Any],
    module_excerpts: list[dict[str, Any]],
    classifier_label: dict[str, Any] | None = None,
    max_tokens: int,
) -> dict[str, Any]:
    module_catalog = "\n".join(f"- {mid}: {name}" for mid, name in AA_MODULES.items())
    snippet_text = "\n\n".join(
        f"[{s['position']}] file: {s['file']}\n```\n{s['excerpt']}\n```"
        for s in sequence["snippets"]
    )
    system = (
        "You write multiple-choice quiz items from ordered code snippets. The goal is "
        "to train an assurance compiler to infer a repo-level assurance thought from a "
        "sequence, while distinguishing plausible but wrong course concepts. Return JSON only."
    )
    user = (
        "Canonical modules:\n"
        f"{module_catalog}\n\n"
        f"Repo: {sequence['repo']}\nSequence id: {sequence['sequence_id']}\n\n"
        "Ordered snippets:\n"
        f"{snippet_text}\n\n"
        "Retrieved course passages. Every answer choice must cite only one of these citations verbatim:\n"
        f"{json.dumps(module_excerpts, ensure_ascii=False, indent=2)}\n\n"
        "Prior classifier label. The correct answer should align with this label unless it is weak or contradicted by retrieved evidence:\n"
        f"{json.dumps(classifier_label or {}, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with exactly these fields:\n"
        "{\n"
        '  "question_type": "module_classification | risk_control | overclaim_boundary | trust_transfer | hidden_witness",\n'
        '  "question": "...",\n'
        '  "choices": [\n'
        '    {"label": "A", "answer": "...", "module": "M12", "citation": "[M12 / file.pdf, p. 1]", "is_correct": true, "why": "..."},\n'
        '    {"label": "B", "answer": "...", "module": "M05", "citation": "[M05 / file.pdf, p. 2]", "is_correct": false, "why": "..."}\n'
        "  ],\n"
        '  "correct_label": "A",\n'
        '  "sequence_thought": "what the ordered snippets imply",\n'
        '  "explanation": "...",\n'
        '  "hard_negative_logic": "why the wrong choices are plausible but wrong",\n'
        '  "risk_portfolio_lesson": "...",\n'
        '  "confidence": "low | medium | high",\n'
        '  "rag_support_status": "supported | weak"\n'
        "}\n"
        "Rules:\n"
        "- Create exactly four choices A-D.\n"
        "- Exactly one choice must be correct.\n"
        "- Each choice must cite a retrieved citation verbatim. Do not invent citations.\n"
        "- Use the sequence order when forming the question.\n"
        "- Prefer questions that expose an overclaim boundary, risk-control distinction, or trust-transfer boundary.\n"
        "- If rag_support_status in the classifier label is inferred_missing_primary_passage, it is acceptable to ask an overclaim-boundary question where the correct answer says the primary claim is inferred rather than directly retrieved.\n"
    )
    raw = _complete_with_retry(
        llm,
        system,
        user,
        temperature=0.1,
        max_tokens=max_tokens,
        retries=getattr(llm, "_rapid_retries", 0),
        sleep_seconds=getattr(llm, "_rapid_sleep_seconds", 10.0),
    )
    quiz = _json_from_text(raw)
    quiz.update({"raw_model_output": raw, "status": "complete" if quiz.get("question") else "failed"})
    return quiz


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Rapid Classifier Run", ""]
    for i, row in enumerate(rows, 1):
        ex = row["example"]
        label = row["label"]
        lines.append(f"## {i}. {ex['repo']} / {ex['file']}")
        lines.append(f"- Primary: `{label.get('primary_module', '?')}`")
        lines.append(f"- Secondary: {', '.join(label.get('secondary_modules', []) or [])}")
        lines.append(f"- Role: {label.get('assurance_role', '')}")
        lines.append(f"- Confidence: {label.get('confidence', '')}")
        if label.get("rag_support_status"):
            lines.append(f"- RAG support: {label.get('rag_support_status')}")
        lines.append(f"- Hidden witness: {label.get('hidden_witness', '')}")
        lines.append(f"- CPS escalation: {label.get('cyberphysical_escalation_point', '')}")
        if label.get("missing_evidence"):
            lines.append("- Missing evidence:")
            for item in label["missing_evidence"][:5]:
                lines.append(f"  - {item}")
        if label.get("overclaim_warning"):
            lines.append(f"- Overclaim: {label['overclaim_warning']}")
        risk = label.get("risk_portfolio") or {}
        if isinstance(risk, dict) and risk.get("portfolio_role"):
            lines.append(f"- Risk role: {risk.get('portfolio_role')}")
        if label.get("rationale"):
            lines.append(f"- Rationale: {label['rationale']}")
        if row.get("quiz"):
            quiz = row["quiz"]
            lines.append("")
            lines.append("### Quiz")
            lines.append(f"**{quiz.get('question', '')}**")
            for choice in quiz.get("choices", []) or []:
                mark = " [correct]" if choice.get("is_correct") else ""
                lines.append(
                    f"- {choice.get('label')}. {choice.get('answer')} "
                    f"| {choice.get('citation', '')}{mark}"
                )
            if quiz.get("explanation"):
                lines.append(f"- Explanation: {quiz['explanation']}")
        lines.append("")
    return "\n".join(lines)


def render_sequence_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Rapid Sequence Classifier Run", ""]
    for i, row in enumerate(rows, 1):
        seq = row["sequence"]
        label = row["label"]
        lines.append(f"## {i}. {seq['repo']} / {seq['sequence_id']}")
        lines.append(f"- Primary: `{label.get('primary_module', '?')}`")
        lines.append(f"- Secondary: {', '.join(label.get('secondary_modules', []) or [])}")
        lines.append(f"- Role: {label.get('repo_assurance_role', '')}")
        lines.append(f"- Confidence: {label.get('confidence', '')}")
        lines.append(f"- RAG support: {label.get('rag_support_status', '')}")
        if label.get("sequence_thought"):
            lines.append(f"- Sequence thought: {label['sequence_thought']}")
        if label.get("hidden_witnesses"):
            lines.append("- Hidden witnesses:")
            for item in label["hidden_witnesses"][:5]:
                lines.append(f"  - {item}")
        if label.get("cyberphysical_escalation_points"):
            lines.append("- CPS escalation:")
            for item in label["cyberphysical_escalation_points"][:5]:
                lines.append(f"  - {item}")
        if label.get("missing_evidence"):
            lines.append("- Missing evidence:")
            for item in label["missing_evidence"][:5]:
                lines.append(f"  - {item}")
        if label.get("overclaim_warnings"):
            lines.append("- Overclaim warnings:")
            for item in label["overclaim_warnings"][:5]:
                lines.append(f"  - {item}")
        risk = label.get("risk_portfolio") or {}
        if isinstance(risk, dict):
            if risk.get("portfolio_role"):
                lines.append(f"- Risk role: {risk.get('portfolio_role')}")
            if risk.get("risk_factors"):
                lines.append("- Risk factors:")
                for item in risk["risk_factors"][:5]:
                    lines.append(f"  - {item}")
            if risk.get("controls_or_hedges"):
                lines.append("- Controls / hedges:")
                for item in risk["controls_or_hedges"][:5]:
                    lines.append(f"  - {item}")
        if row.get("quiz"):
            quiz = row["quiz"]
            lines.append("")
            lines.append("### Quiz")
            lines.append(f"**{quiz.get('question', '')}**")
            for choice in quiz.get("choices", []) or []:
                mark = " [correct]" if choice.get("is_correct") else ""
                lines.append(
                    f"- {choice.get('label')}. {choice.get('answer')} "
                    f"| {choice.get('citation', '')}{mark}"
                )
            if quiz.get("explanation"):
                lines.append(f"- Explanation: {quiz['explanation']}")
        lines.append("")
    return "\n".join(lines)


def _fold_id(repo: str, folds: int) -> int:
    if folds <= 1:
        return 0
    return sum(ord(c) for c in repo.lower()) % folds


def run(args: argparse.Namespace) -> int:
    if args.repo:
        repos = [Path(p) for p in args.repo]
    else:
        repos = sorted(p for p in Path(args.repo_root).iterdir() if p.is_dir())
    if args.max_repos:
        repos = repos[: args.max_repos]
    index = RagIndex.load(args.index_dir)
    llm = get_llm(args.provider, model=args.model)
    setattr(llm, "_rapid_retries", args.retries)
    setattr(llm, "_rapid_sleep_seconds", args.sleep_seconds)
    mode_label = "sequence" if args.sequence_mode else "snippet"
    out_dir = Path(args.out_dir) / f"rapid_classifier_{mode_label}_{args.provider}_{(args.model or 'default').replace('.', '-').replace(':', '-')}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "classified_examples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        if args.sequence_mode:
            sequences = sample_repo_sequences(
                repos,
                sequences_per_repo=args.sequences_per_repo,
                snippets_per_sequence=args.snippets_per_sequence,
                max_chars=args.excerpt_chars,
                seed=args.seed,
            )
            if args.max_examples:
                sequences = sequences[: args.max_examples]
            for i, seq in enumerate(sequences, 1):
                _safe_print(f"[{i}/{len(sequences)}] classify sequence {seq['repo']} / {seq['sequence_id']}")
                retrieval_query = seq["repo"] + "\n" + "\n".join(s["excerpt"][:800] for s in seq["snippets"])
                passages = retrieve_module_excerpts(index, retrieval_query, k=args.rag_k)
                label = classify_sequence(
                    llm=llm,
                    sequence=seq,
                    module_excerpts=passages,
                    max_tokens=args.max_tokens,
                )
                quiz = None
                if args.quiz:
                    quiz = generate_quiz_for_sequence(
                        llm=llm,
                        sequence=seq,
                        module_excerpts=passages,
                        classifier_label=label,
                        max_tokens=args.quiz_max_tokens,
                    )
                row = {
                    "fold_id": _fold_id(seq["repo"], args.folds),
                    "sequence": seq,
                    "module_excerpts": passages,
                    "label": label,
                }
                if quiz is not None:
                    row["quiz"] = quiz
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        else:
            examples = sample_repo_excerpts(
                repos,
                samples_per_repo=args.samples_per_repo,
                max_chars=args.excerpt_chars,
                seed=args.seed,
            )
            if args.max_examples:
                examples = examples[: args.max_examples]
            for i, ex in enumerate(examples, 1):
                _safe_print(f"[{i}/{len(examples)}] classify {ex['repo']} / {ex['file']}")
                retrieval_query = f"{ex['repo']} {ex['file']}\n{ex['excerpt'][:1800]}"
                passages = retrieve_module_excerpts(index, retrieval_query, k=args.rag_k)
                label = classify_excerpt(
                    llm=llm,
                    example=ex,
                    module_excerpts=passages,
                    max_tokens=args.max_tokens,
                )
                quiz = None
                if args.quiz:
                    quiz = generate_quiz_for_excerpt(
                        llm=llm,
                        example=ex,
                        module_excerpts=passages,
                        classifier_label=label,
                        max_tokens=args.quiz_max_tokens,
                    )
                row = {
                    "fold_id": _fold_id(ex["repo"], args.folds),
                    "example": ex,
                    "module_excerpts": passages,
                    "label": label,
                }
                if quiz is not None:
                    row["quiz"] = quiz
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    md = render_sequence_markdown(rows) if args.sequence_mode else render_markdown(rows)
    (out_dir / "rapid_classifier.md").write_text(md, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({
            "provider": args.provider,
            "model": args.model,
            "mode": mode_label,
            "repo_count": len(repos),
            "example_count": len(rows),
            "folds": args.folds,
            "quiz": args.quiz,
            "jsonl": str(jsonl_path),
        }, indent=2),
        encoding="utf-8",
    )
    _safe_print(f"Saved rapid classifier directory: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--repo", action="append", help="Specific repo path. Repeatable.")
    parser.add_argument("--index-dir", default=".aate_index")
    parser.add_argument("--provider", default="mistral",
                        choices=["mistral", "openai", "xai", "grok", "anthropic", "ollama", "stub", "peft-local"])
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--samples-per-repo", type=int, default=2)
    parser.add_argument("--sequence-mode", action="store_true",
                        help="Classify ordered repo-level sequences rather than isolated snippets.")
    parser.add_argument("--sequences-per-repo", type=int, default=1)
    parser.add_argument("--snippets-per-sequence", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5,
                        help="Assign deterministic repo-level fold IDs for cross-validation.")
    parser.add_argument("--quiz", action="store_true",
                        help="Also generate a citation-grounded multiple-choice quiz item.")
    parser.add_argument("--quiz-max-tokens", type=int, default=1700)
    parser.add_argument("--max-repos", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--excerpt-chars", type=int, default=2600)
    parser.add_argument("--rag-k", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

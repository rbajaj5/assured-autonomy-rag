"""Debate Assured Autonomy concepts before applying them to outside policy facts."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from .hybrid_retriever import HybridPaperRetriever, HybridRetrievalConfig
from .indexing import Hit, RagIndex
from .llm import get_llm
from .paths import DEFAULT_INDEX_DIR, PAPERS_DIR
from .policy_rag import _expand_policy_source_refs, build_multi_policy_prompt, load_policy_sources


@dataclass
class ConceptDebateConfig:
    index_dir: Path = DEFAULT_INDEX_DIR
    papers_dir: Path = PAPERS_DIR
    provider_a: str = "openai"
    model_a: str = "gpt-4.1"
    provider_b: str = "mistral"
    model_b: str = "mistral-large-latest"
    judge_provider: str = "openai"
    judge_model: str = "gpt-4.1"
    top_k: int = 4
    max_tokens: int = 5000
    stage_retries: int = 2
    out_dir: Path = Path("concept_runs")


DEFAULT_CONCEPT_QUESTION = (
    "How should Assured Autonomy reason about access to a restricted academic dataset? "
    "Debate tensions among permission governance, privacy/data minimization, accountability, "
    "human oversight, documentation/traceability, operational constraints, and research utility."
)


CONCEPT_QUERIES = [
    (
        "assured_autonomy_framing",
        "assured autonomy trustworthy autonomous systems risk governance assurance lifecycle",
        ["M01-02"],
    ),
    (
        "safety_security_privacy",
        "security privacy threat model attacker capabilities confidentiality integrity availability personal data",
        ["M03"],
    ),
    (
        "anomaly_false_positives",
        "anomaly detection outlier detection false positives fraud intrusion insider threat detection threshold",
        ["M04"],
    ),
    (
        "dataset_shift_base_rate",
        "dataset shift distribution drift base rate nonstationary environment detection false alarms",
        ["M05"],
    ),
    (
        "verification_validation",
        "verification validation specification assurance claims evidence thresholds correctness safety certification",
        ["M06"],
    ),
    (
        "test_evaluation",
        "test evaluation operational testing assurance metrics validation deployment monitoring",
        ["M07"],
    ),
    (
        "interpretability_contestability",
        "interpretability explainability explanation contestability transparency human decision support",
        ["M08"],
    ),
    (
        "human_oversight_accountability",
        "human oversight accountability responsibility automation bias human in the loop stakeholder governance",
        ["M09"],
    ),
    (
        "policy_governance",
        "governance accountability authorization permission policy responsible AI auditability data governance",
        ["M10"],
    ),
    (
        "adversarial_autonomy",
        "adversarial autonomy evasion attack deception threat model malicious actor robustness security",
        ["M11"],
    ),
    (
        "runtime_assurance",
        "runtime assurance monitoring safe fallback switching safety envelope untrusted component intervention",
        ["M12"],
    ),
    (
        "multi_agent_cooperation",
        "multi agent cooperation coordination distributed agents trust accountability remote workers organizations",
        ["M14"],
    ),
]


def run_concept_debate(cfg: ConceptDebateConfig, question: str = DEFAULT_CONCEPT_QUESTION) -> dict:
    run_dir, stem = _new_concept_run_dir(cfg.out_dir)
    evidence = _retrieve_concept_evidence(cfg, question)
    evidence_block = _evidence_prompt(evidence)
    a = get_llm(cfg.provider_a, model=cfg.model_a)
    b = get_llm(cfg.provider_b, model=cfg.model_b)
    judge = get_llm(cfg.judge_provider, model=cfg.judge_model)

    system = (
        "You are debating interpretations of Assured Autonomy course readings. "
        "Use only the provided AA evidence handles [AA1], [AA2], etc. Do not discuss "
        "JuReko or any external policy yet."
    )
    prompt = (
        f"Concept debate question:\n{question}\n\n"
        f"Assured Autonomy evidence:\n{evidence_block}\n\n"
        "Extract concepts as reusable reasoning vectors. For each concept, explain "
        "what the readings support, what they do not settle, and what tensions exist."
    )
    result = {
        "question": question,
        "evidence": evidence,
        "debater_a": {"provider": cfg.provider_a, "model": cfg.model_a},
        "debater_b": {"provider": cfg.provider_b, "model": cfg.model_b},
        "judge": {"provider": cfg.judge_provider, "model": cfg.judge_model},
        "a_open": "",
        "b_open": "",
        "a_rebuttal": "",
        "b_rebuttal": "",
        "judgment": "",
        "status": "started",
    }
    _save_concept_progress(result, run_dir, stem)

    a_open = _call_stage(
        lambda budget: a.complete(
            system,
            prompt + "\n\nRole A: conservative assurance/governance interpretation. Emphasize permission, accountability, and constraints.",
            temperature=0.2,
            max_tokens=budget,
        ),
        stage="a_open",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["a_open"] = a_open
    result["status"] = "a_open_complete"
    _save_concept_progress(result, run_dir, stem)

    b_open = _call_stage(
        lambda budget: b.complete(
            system,
            prompt + "\n\nRole B: risk-managed research enablement interpretation. Emphasize minimization, oversight, and constrained utility.",
            temperature=0.2,
            max_tokens=budget,
        ),
        stage="b_open",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["b_open"] = b_open
    result["status"] = "b_open_complete"
    _save_concept_progress(result, run_dir, stem)

    a_rebuttal = _call_stage(
        lambda budget: a.complete(
            system,
            f"{prompt}\n\nA opening:\n{a_open}\n\nB opening:\n{b_open}\n\nRole A rebuttal: identify where B overextends the readings and where B is useful.",
            temperature=0.2,
            max_tokens=budget,
        ),
        stage="a_rebuttal",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["a_rebuttal"] = a_rebuttal
    result["status"] = "a_rebuttal_complete"
    _save_concept_progress(result, run_dir, stem)

    b_rebuttal = _call_stage(
        lambda budget: b.complete(
            system,
            f"{prompt}\n\nA opening:\n{a_open}\n\nB opening:\n{b_open}\n\nRole B rebuttal: identify where A overconstrains the readings and where A is useful.",
            temperature=0.2,
            max_tokens=budget,
        ),
        stage="b_rebuttal",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["b_rebuttal"] = b_rebuttal
    result["status"] = "b_rebuttal_complete"
    _save_concept_progress(result, run_dir, stem)

    judgment = _call_stage(
        lambda budget: judge.complete(
            "You are a neutral judge building an Assured Autonomy concept map. Use [AA] citations.",
            (
                f"{prompt}\n\nA opening:\n{a_open}\n\nB opening:\n{b_open}\n\n"
                f"A rebuttal:\n{a_rebuttal}\n\nB rebuttal:\n{b_rebuttal}\n\n"
                "Produce: 1) concept map table, 2) genuine tensions in the readings, "
                "3) dominant concept vectors, 4) weak analogies to avoid, 5) a JSON block "
                "called CONCEPT_MAP_JSON with concept ids and short descriptions."
            ),
            temperature=0.1,
            max_tokens=budget,
        ),
        stage="judge",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["judgment"] = judgment
    result["status"] = "complete"
    _save_concept_run(result, run_dir, stem)
    return result


def apply_concepts_to_policy(*, concept_run: Path, policy_urls: list[str] | None = None,
                             policy_file: str | None = None, policy_text: str | None = None,
                             provider: str = "openai", model: str = "gpt-4.1",
                             question: str, max_policy_chars: int = 60000,
                             max_tokens: int = 5000,
                             out_dir: Path = Path("concept_runs")) -> dict:
    concept = json.loads(concept_run.read_text(encoding="utf-8"))
    items = load_policy_sources(policy_urls=policy_urls, policy_file=policy_file, policy_text=policy_text)
    policy_prompt = build_multi_policy_prompt(items, question, max_policy_chars=max_policy_chars)
    llm = get_llm(provider, model=model)
    answer = llm.complete(
        "Apply an Assured Autonomy concept map to supplied policy/source facts. Keep concept claims separate from source facts.",
        (
            f"Concept debate judgment:\n{concept['judgment']}\n\n"
            f"Concept evidence metadata:\n{json.dumps(concept['evidence'], indent=2)}\n\n"
            f"Policy/source prompt:\n{policy_prompt}\n\n"
            "Apply the concept map to the policy question. Produce: 1) source facts, "
            "2) activated Assured Autonomy concepts, 3) concept tensions, 4) final access strategy, "
            "5) overclaims to avoid, 6) questions/email."
        ),
        temperature=0.1,
        max_tokens=max_tokens,
    )
    answer = _expand_policy_source_refs(answer, items)
    result = {
        "concept_run": str(concept_run),
        "policy_source": [{"source": i.source, "title": i.title, "source_kind": i.source_kind} for i in items],
        "provider": provider,
        "model": model,
        "question": question,
        "answer": answer,
    }
    _save_apply_run(result, out_dir)
    return result


def _retrieve_concept_evidence(cfg: ConceptDebateConfig, question: str) -> list[dict]:
    index = RagIndex.load(cfg.index_dir)
    retriever = HybridPaperRetriever(
        index,
        corpus_dir=cfg.papers_dir,
        cfg=HybridRetrievalConfig(top_k=cfg.top_k),
    )
    evidence = []
    seen = set()
    per_seed_k = max(1, min(cfg.top_k, 3))
    for concept, query, modules in CONCEPT_QUERIES:
        retrieval_query = (
            f"{query}\n\n"
            f"Case context, used only for relevance:\n{question}"
        )
        for hit in retriever.retrieve(retrieval_query, top_k=per_seed_k, modules=modules):
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            evidence.append({
                "handle": f"AA{len(evidence) + 1}",
                "concept_seed": concept,
                "source": _source(hit),
                "module": hit.chunk.module_name,
                "path": str(cfg.papers_dir / hit.chunk.relpath),
                "text": hit.chunk.text[:1200],
            })
    return evidence


def _evidence_prompt(evidence: list[dict]) -> str:
    return "\n\n".join(
        f"[{e['handle']}] Seed: {e['concept_seed']}; Source: {e['source']}; Module: {e['module']}; Local Path: {e['path']}\n{e['text']}"
        for e in evidence
    )


def _source(hit: Hit) -> str:
    parts = Path(hit.chunk.filename).stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return Path(hit.chunk.filename).stem


def _call_stage(call, *, stage: str, max_tokens: int, retries: int) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        budget = max(1200, int(max_tokens * (0.7 ** attempt)))
        try:
            return call(budget)
        except Exception as exc:
            last_exc = exc
            print(f"  ! {stage} failed on attempt {attempt + 1} with {budget} tokens: {exc}")
    raise RuntimeError(f"{stage} failed after {retries + 1} attempts") from last_exc


def _new_concept_run_dir(out_dir: Path) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = out_dir / f"concept_debate_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    stem = run_dir / f"concept_debate_{stamp}"
    return run_dir, stem


def _save_concept_progress(result: dict, run_dir: Path, stem: Path) -> None:
    Path(f"{stem}.partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(f"{stem}.partial.md").write_text(_render_concept_md(result), encoding="utf-8")
    print(f"Wrote progress {Path(f'{stem}.partial.json')}")


def _save_concept_run(result: dict, run_dir: Path, stem: Path) -> None:
    Path(f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(f"{stem}.md").write_text(_render_concept_md(result), encoding="utf-8")
    print(f"Saved concept debate directory: {run_dir}")


def _save_apply_run(result: dict, out_dir: Path) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = out_dir / f"concept_apply_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    stem = run_dir / f"concept_apply_{stamp}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(_render_apply_md(result), encoding="utf-8")
    print(f"Saved concept application directory: {run_dir}")


def _render_concept_md(result: dict) -> str:
    return "\n".join([
        "# Concept Debate",
        "",
        "## Question",
        result["question"],
        "",
        "## Evidence",
        "```json",
        json.dumps(result["evidence"], indent=2),
        "```",
        "",
        "## A Opening",
        result["a_open"],
        "",
        "## B Opening",
        result["b_open"],
        "",
        "## A Rebuttal",
        result["a_rebuttal"],
        "",
        "## B Rebuttal",
        result["b_rebuttal"],
        "",
        "## Judgment",
        result["judgment"],
        "",
    ])


def _render_apply_md(result: dict) -> str:
    return "\n".join([
        "# Concept Application",
        "",
        f"Concept run: {result['concept_run']}",
        "",
        "## Policy Sources",
        "```json",
        json.dumps(result["policy_source"], indent=2),
        "```",
        "",
        "## Answer",
        result["answer"],
        "",
    ])

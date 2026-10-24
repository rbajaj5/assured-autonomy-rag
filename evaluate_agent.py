"""Evaluation harness for the agentic Assured Autonomy RAG assistant."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .agentic_rag import AgenticRagAgent, AgenticRagConfig


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected_modules: tuple[str, ...]


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "m12_vs_m06",
        "How is Runtime Assurance different from Formal Verification?",
        ("Runtime Assurance", "Verification and Validation"),
    ),
    EvalCase(
        "m05_vs_m04",
        "Compare Dataset Shift and Anomaly Detection for autonomous systems.",
        ("Dataset Shift", "Anomaly and Fault Detection"),
    ),
    EvalCase(
        "m12_plus_m13",
        "How does Runtime Assurance complement Safe Reinforcement Learning?",
        ("Runtime Assurance", "Safe Reinforcement Learning"),
    ),
    EvalCase(
        "m11_vs_m03",
        "How does Adversarial Autonomy differ from Safety Security and Privacy in this course?",
        ("Adversarial Autonomy", "Safety, Security, and Privacy"),
    ),
    EvalCase(
        "m08_vs_m09",
        "Compare Interpretable Autonomy and Human-AI Integration for deployed autonomous systems.",
        ("Interpretable Autonomy", "Human-AI Integration"),
    ),
)


ADVANCED_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "fog_construction_perception_failure",
        (
            "An autonomous vehicle perception model performs well in simulation but "
            "fails after deployment in fog and construction zones. Classify the "
            "assurance problem and recommend a layered response."
        ),
        ("Dataset Shift", "Anomaly and Fault Detection", "Runtime Assurance", "Test and Evaluation"),
    ),
    EvalCase(
        "adversarial_patch_vehicle_stack",
        (
            "A sticker-like adversarial patch causes an autonomous vehicle perception "
            "stack to misclassify stop signs. Map mitigations across security, "
            "adversarial robustness, testing, and runtime assurance."
        ),
        ("Safety, Security, and Privacy", "Adversarial Autonomy", "Test and Evaluation", "Runtime Assurance"),
    ),
    EvalCase(
        "black_box_controller_assurance_case",
        (
            "You inherit a black-box ML controller for a drone. Compare formal "
            "verification, testing, runtime monitoring, interpretability, and human "
            "oversight as assurance layers, noting what each cannot guarantee."
        ),
        (
            "Verification and Validation",
            "Test and Evaluation",
            "Runtime Assurance",
            "Interpretable Autonomy",
            "Human-AI Integration",
        ),
    ),
    EvalCase(
        "safe_rl_runtime_handoff",
        (
            "Design a safety case for a Safe RL navigation policy that may encounter "
            "out-of-distribution states. Explain when Runtime Assurance should override "
            "the learned policy and when human oversight is needed."
        ),
        ("Safe Reinforcement Learning", "Runtime Assurance", "Dataset Shift", "Human-AI Integration"),
    ),
    EvalCase(
        "interpretability_not_oversight",
        (
            "Explain why interpretability does not automatically solve human-AI "
            "integration problems in deployed autonomous systems, and identify what "
            "governance or operational controls are still needed."
        ),
        ("Interpretable Autonomy", "Human-AI Integration", "Policy and Governance"),
    ),
)


EVAL_SUITES: dict[str, tuple[EvalCase, ...]] = {
    "basic": DEFAULT_CASES,
    "advanced": ADVANCED_CASES,
    "all": DEFAULT_CASES + ADVANCED_CASES,
}


_CITATION_RE = re.compile(
    r"Source:\s*(?P<source>[^|\n]+)\|\s*Module:\s*(?P<module>[^|\n]+)\|\s*Local Path:\s*(?P<path>[^\]\n.]+\.pdf)",
    flags=re.IGNORECASE,
)


def _extract_citations(text: str) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for match in _CITATION_RE.finditer(text):
        citations.append({
            "source": match.group("source").strip(),
            "module": match.group("module").strip(),
            "path": match.group("path").strip(),
        })
    return citations


def _score_answer(answer: str, expected_modules: tuple[str, ...]) -> dict:
    citations = _extract_citations(answer)
    cited_modules = {c["module"] for c in citations}
    words = re.findall(r"\b\w+\b", answer)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer)
        if len(s.strip()) > 20
    ]
    cited_sentences = [s for s in sentences if "Source:" in s and "Local Path:" in s]
    module_counts = {
        module: sum(1 for c in citations if module.lower() in c["module"].lower())
        for module in expected_modules
    }
    nonzero_counts = [count for count in module_counts.values() if count > 0]
    module_balance = (
        min(nonzero_counts) / max(nonzero_counts)
        if len(nonzero_counts) == len(expected_modules) and max(nonzero_counts) > 0
        else 0.0
    )
    citation_density = len(citations) / max(len(words) / 100.0, 1.0)
    cited_sentence_ratio = len(cited_sentences) / max(len(sentences), 1)
    length_score = 1.0 if 120 <= len(words) <= 900 else max(0.0, 1.0 - abs(len(words) - 450) / 900)
    density_score = min(citation_density / 1.5, 1.0)
    unsupported_risk = 1.0 - cited_sentence_ratio
    quality_score = round(100 * (
        0.30 * min(cited_sentence_ratio, 1.0)
        + 0.25 * module_balance
        + 0.20 * density_score
        + 0.15 * length_score
        + 0.10 * (1.0 - unsupported_risk)
    ))
    path_checks = [
        {
            "path": c["path"],
            "exists": Path(c["path"]).exists(),
        }
        for c in citations
    ]
    expected_present = {
        module: any(module.lower() in cited.lower() for cited in cited_modules)
        for module in expected_modules
    }
    return {
        "expected_modules": list(expected_modules),
        "cited_modules": sorted(cited_modules),
        "n_citations": len(citations),
        "all_expected_modules_cited": all(expected_present.values()),
        "expected_module_presence": expected_present,
        "all_cited_paths_exist": all(item["exists"] for item in path_checks) if path_checks else False,
        "path_checks": path_checks,
        "citations": citations,
        "quality": {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "cited_sentence_count": len(cited_sentences),
            "citation_density_per_100_words": round(citation_density, 2),
            "cited_sentence_ratio": round(cited_sentence_ratio, 2),
            "module_citation_counts": module_counts,
            "module_balance": round(module_balance, 2),
            "unsupported_claim_risk": round(unsupported_risk, 2),
            "quality_score_0_100": quality_score,
        },
    }


def _render_markdown(rows: list[dict]) -> str:
    lines = ["# Agentic RAG Evaluation", ""]
    passed = sum(1 for row in rows if row["pass"])
    lines.append(f"Passed {passed}/{len(rows)} cases.")
    lines.append("")
    lines.append("| Case | Pass | Quality | Words | Citations | Density | Balance | Unsupported Risk |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        score = row["score"]
        quality = score.get("quality", {})
        lines.append(
            f"| {row['case_id']} | {'yes' if row['pass'] else 'no'} | "
            f"{quality.get('quality_score_0_100', 0)} | "
            f"{quality.get('word_count', 0)} | "
            f"{score['n_citations']} | "
            f"{quality.get('citation_density_per_100_words', 0)} | "
            f"{quality.get('module_balance', 0)} | "
            f"{quality.get('unsupported_claim_risk', 0)} |"
        )
    for row in rows:
        lines.extend([
            "",
            f"## {row['case_id']}",
            "",
            f"**Query:** {row['query']}",
            "",
            f"**Pass:** {'yes' if row['pass'] else 'no'}",
            "",
            f"**Quality:** {row.get('score', {}).get('quality', {})}",
            "",
            "```text",
            row.get("answer", row.get("error", "")),
            "```",
        ])
    return "\n".join(lines) + "\n"


def run_eval(cfg: AgenticRagConfig, *, out_dir: str | Path = "eval_runs",
             cases: tuple[EvalCase, ...] = DEFAULT_CASES) -> Path:
    agent = AgenticRagAgent(cfg)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = out_root / f"agent_eval_{cfg.provider}_{cfg.model or 'default'}_{stamp}"
    rows: list[dict] = []
    for case in cases:
        row = asdict(case)
        try:
            answer = agent.answer(case.query)
            score = _score_answer(answer, case.expected_modules)
            row.update({
                "answer": answer,
                "score": score,
                "pass": score["all_expected_modules_cited"] and score["all_cited_paths_exist"],
            })
            print(
                f"[{case.case_id}] pass={row['pass']} "
                f"citations={score['n_citations']} modules={score['cited_modules']}"
            )
        except Exception as exc:
            row.update({
                "error": repr(exc),
                "score": {},
                "pass": False,
            })
            print(f"[{case.case_id}] ERROR {exc}")
        rows.append(row)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return json_path


def rescore_report(path: str | Path) -> Path:
    src = Path(path)
    rows = json.loads(src.read_text(encoding="utf-8"))
    for row in rows:
        if "answer" not in row:
            continue
        expected = tuple(row.get("expected_modules", ()))
        score = _score_answer(row["answer"], expected)
        row["score"] = score
        row["pass"] = score["all_expected_modules_cited"] and score["all_cited_paths_exist"]
    src.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path = src.with_suffix(".md")
    md_path.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"Rescored {src}")
    print(f"Wrote {md_path}")
    return src


def compare_reports(paths: list[str | Path], *, out_path: str | Path | None = None) -> Path:
    summaries: list[dict] = []
    for path in paths:
        src = Path(path)
        rows = json.loads(src.read_text(encoding="utf-8"))
        scores = [row.get("score", {}).get("quality", {}).get("quality_score_0_100", 0) for row in rows]
        citations = [row.get("score", {}).get("n_citations", 0) for row in rows]
        pass_count = sum(1 for row in rows if row.get("pass"))
        summaries.append({
            "report": str(src),
            "cases": len(rows),
            "pass_count": pass_count,
            "avg_quality": round(sum(scores) / max(len(scores), 1), 1),
            "avg_citations": round(sum(citations) / max(len(citations), 1), 1),
            "avg_words": round(sum(row.get("score", {}).get("quality", {}).get("word_count", 0) for row in rows) / max(len(rows), 1), 1),
            "avg_unsupported_risk": round(sum(row.get("score", {}).get("quality", {}).get("unsupported_claim_risk", 0) for row in rows) / max(len(rows), 1), 2),
        })
    if out_path is None:
        out_path = Path(paths[0]).parent / "agent_eval_comparison.md"
    out = Path(out_path)
    lines = ["# Agent Evaluation Comparison", ""]
    lines.append("| Report | Pass | Avg Quality | Avg Citations | Avg Words | Avg Unsupported Risk |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summaries:
        lines.append(
            f"| {Path(item['report']).name} | {item['pass_count']}/{item['cases']} | "
            f"{item['avg_quality']} | {item['avg_citations']} | "
            f"{item['avg_words']} | {item['avg_unsupported_risk']} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return out

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api_keys import get_api_key
from .external_judge import deterministic_fields_unchanged, run_external_judge
from .pipeline import evaluate_proposal


VERDICTS = ["approve", "revise", "reject", "unknown"]
RISKS = ["low", "medium", "high", "unacceptable", "unknown"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Mistral external-judge label stability on one proposal fixture.")
    parser.add_argument("proposal_file", help="Existing .txt proposal fixture.")
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--allow-api-call", action="store_true", help="Permit real Mistral API calls.")
    parser.add_argument("--model", help="Mistral model. Defaults to ASSURED_RAG_MISTRAL_MODEL or project default.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--summary", required=True, help="Markdown summary output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_api_call and os.environ.get("ASSURED_RAG_ALLOW_API_CALLS") != "1":
        raise SystemExit("Real Mistral stability calls require --allow-api-call or ASSURED_RAG_ALLOW_API_CALLS=1.")
    if not get_api_key("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY is missing; real-call stability experiment skipped.")
    model = args.model or os.environ.get("ASSURED_RAG_MISTRAL_MODEL") or "mistral-large-latest"
    result = run_stability_experiment(
        proposal_file=Path(args.proposal_file),
        calls=args.calls,
        model=model,
        allow_api_call=True,
    )
    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(render_summary(result), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {summary}")
    return 0


def run_stability_experiment(
    *,
    proposal_file: Path,
    calls: int,
    model: str,
    allow_api_call: bool,
) -> dict[str, Any]:
    proposal_text = proposal_file.read_text(encoding="utf-8")
    deterministic = evaluate_proposal(proposal_text, citations_k=3, use_embeddings=False)
    deterministic_before = deterministic.to_dict()
    rows = []
    for call_index in range(1, calls + 1):
        rows.append(_run_one_stability_call(
            call_index=call_index,
            proposal_text=proposal_text,
            deterministic_result=deterministic_before,
            model=model,
            allow_api_call=allow_api_call,
        ))
    metrics = compute_stability_metrics(rows, deterministic_before)
    return {
        "title": "Mistral Label-Stability Experiment",
        "proposal_file": str(proposal_file),
        "fixture_name": proposal_file.name,
        "model": model,
        "calls_requested": calls,
        "calls_completed": len(rows),
        "deterministic": {
            "verdict": deterministic_before.get("verdict"),
            "risk_level": deterministic_before.get("risk_level"),
            "final_rule": deterministic_before.get("decision_trace", {}).get("final_rule", ""),
        },
        "rows": rows,
        "metrics": metrics,
        "interpretation": interpret_stability(metrics, deterministic_before),
    }


def _run_one_stability_call(
    *,
    call_index: int,
    proposal_text: str,
    deterministic_result: dict[str, Any],
    model: str,
    allow_api_call: bool,
) -> dict[str, Any]:
    judge = run_external_judge(
        proposal_text,
        deterministic_result,
        provider="mistral",
        model=model,
        allow_api_call=allow_api_call,
    )
    after = dict(deterministic_result)
    after["external_judge"] = judge
    unchanged = deterministic_fields_unchanged(deterministic_result, after)
    judge["external_judge_non_override_verified"] = unchanged
    return {
        "call_index": call_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deterministic_verdict": deterministic_result.get("verdict"),
        "deterministic_risk_level": deterministic_result.get("risk_level"),
        "deterministic_final_rule": deterministic_result.get("decision_trace", {}).get("final_rule", ""),
        "mistral_verdict_recommendation": judge.get("judge_verdict_recommendation", "unknown"),
        "mistral_risk_recommendation": judge.get("judge_risk_recommendation", "unknown"),
        "judge_status": judge.get("judge_status", "unknown"),
        "parse_strategy": judge.get("parse_strategy", "unknown"),
        "external_judge_non_override_verified": unchanged,
        "deterministic_fields_changed": not unchanged,
        "raw_output_excerpt": judge.get("raw_output_excerpt", ""),
        "error_reason": judge.get("reason", ""),
    }


def compute_stability_metrics(rows: list[dict[str, Any]], deterministic: dict[str, Any]) -> dict[str, Any]:
    verdict_distribution = Counter(row.get("mistral_verdict_recommendation", "unknown") for row in rows)
    risk_distribution = Counter(row.get("mistral_risk_recommendation", "unknown") for row in rows)
    substantive_rows = [
        row for row in rows
        if row.get("judge_status") == "ok"
        and row.get("mistral_verdict_recommendation") != "unknown"
        and row.get("mistral_risk_recommendation") != "unknown"
    ]
    approve_unacceptable = sum(
        1 for row in rows
        if row.get("mistral_verdict_recommendation") == "approve"
        and row.get("mistral_risk_recommendation") == "unacceptable"
    )
    same_input_disagreements = sum(
        1 for row in substantive_rows
        if row.get("mistral_verdict_recommendation") != deterministic.get("verdict")
        or row.get("mistral_risk_recommendation") != deterministic.get("risk_level")
    )
    parse_errors = sum(1 for row in rows if row.get("judge_status") != "ok" or row.get("parse_strategy") in {"failed", "unknown"})
    non_override_failures = sum(1 for row in rows if row.get("deterministic_fields_changed"))
    return {
        "verdict_distribution": dict(verdict_distribution),
        "risk_distribution": dict(risk_distribution),
        "verdict_entropy": entropy(verdict_distribution),
        "risk_entropy": entropy(risk_distribution),
        "approve_unacceptable_frequency": _rate(approve_unacceptable, rows),
        "approve_unacceptable_count": approve_unacceptable,
        "same_input_disagreement_rate": _rate(same_input_disagreements, substantive_rows),
        "same_input_disagreement_count": same_input_disagreements,
        "substantive_call_count": len(substantive_rows),
        "parse_error_rate": _rate(parse_errors, rows),
        "parse_error_count": parse_errors,
        "deterministic_non_override_failures": non_override_failures,
    }


def entropy(distribution: Counter | dict[str, int]) -> float:
    counts = list(distribution.values())
    total = sum(counts)
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count:
            p = count / total
            value -= p * math.log2(p)
    return round(value, 4)


def interpret_stability(metrics: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, str]:
    if metrics.get("deterministic_non_override_failures", 0) > 0:
        return {
            "label": "implementation_bug",
            "summary": "Deterministic non-override failures were observed; treat this as an implementation bug before interpreting Mistral behavior.",
        }
    verdict_dist = metrics.get("verdict_distribution", {})
    risk_dist = metrics.get("risk_distribution", {})
    verdict_entropy = float(metrics.get("verdict_entropy", 0.0))
    risk_entropy = float(metrics.get("risk_entropy", 0.0))
    total = sum(verdict_dist.values())
    approve_unacceptable_count = int(metrics.get("approve_unacceptable_count", 0))
    if total and approve_unacceptable_count / total >= 0.6:
        return {
            "label": "stable_decision_rule_blind_spot",
            "summary": "Mistral repeatedly produced approve/unacceptable, indicating a stable failure to propagate unacceptable risk into the verdict.",
        }
    if risk_dist.get("unacceptable", 0) >= max(1, int(total * 0.6)) and verdict_entropy > 0.5:
        return {
            "label": "verdict_propagation_instability",
            "summary": "Mistral usually recognized unacceptable risk but varied on the verdict, suggesting instability in propagating risk into the approval decision.",
        }
    if verdict_entropy > 0.9 or risk_entropy > 0.9:
        return {
            "label": "label_instability",
            "summary": "Mistral labels varied widely for the same input, suggesting stochastic label instability.",
        }
    if metrics.get("same_input_disagreement_rate", 0.0) > 0.5:
        return {
            "label": "stable_or_low_variance_disagreement",
            "summary": "Mistral consistently disagreed with the deterministic evaluator without broad label variation.",
        }
    return {
        "label": "mostly_stable_alignment",
        "summary": "Mistral labels were relatively stable and mostly aligned for this fixture.",
    }


def render_summary(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    interp = result["interpretation"]
    lines = [
        "# Mistral Label-Stability Experiment",
        "",
        f"- Fixture: `{result['fixture_name']}`",
        f"- Model: `{result['model']}`",
        f"- Calls completed: {result['calls_completed']}",
        f"- Deterministic verdict/risk/final rule: `{result['deterministic']['verdict']}` / `{result['deterministic']['risk_level']}` / `{result['deterministic']['final_rule']}`",
        f"- Interpretation: `{interp['label']}`",
        f"- Interpretation summary: {interp['summary']}",
        f"- Deterministic non-override failures: {metrics['deterministic_non_override_failures']}",
        "",
        "## Metrics",
        "",
        f"- Verdict entropy: {metrics['verdict_entropy']}",
        f"- Risk entropy: {metrics['risk_entropy']}",
        f"- Approve/unacceptable frequency: {metrics['approve_unacceptable_frequency']:.3f}",
        f"- Same-input disagreement rate: {metrics['same_input_disagreement_rate']:.3f}",
        f"- Parse/error rate: {metrics['parse_error_rate']:.3f}",
        "",
        "## Verdict Distribution",
        "",
    ]
    lines.extend(_counter_lines(metrics["verdict_distribution"]))
    lines.extend(["", "## Risk Distribution", ""])
    lines.extend(_counter_lines(metrics["risk_distribution"]))
    return "\n".join(lines).strip() + "\n"


def _counter_lines(distribution: dict[str, int]) -> list[str]:
    if not distribution:
        return ["- none: 0"]
    return [f"- `{key}`: {value}" for key, value in sorted(distribution.items())]


def _rate(count: int, denominator: list[Any]) -> float:
    return round(count / len(denominator), 4) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

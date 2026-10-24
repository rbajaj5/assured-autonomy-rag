from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api_keys import get_api_key
from .external_judge import deterministic_fields_unchanged, run_external_judge
from .pipeline import evaluate_proposal


TRANSIENT_ERROR_TERMS = ["429", "rate", "timeout", "temporar", "connection", "503", "502", "504"]
BACKOFF_SECONDS = [15, 30, 60, 120, 180]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect real Mistral external-judge calls for assurance evaluator testing.")
    parser.add_argument("proposal_dir", help="Directory containing existing .txt proposal fixtures.")
    parser.add_argument("--calls", type=int, required=True, help="Minimum number of rows/calls to collect.")
    parser.add_argument("--allow-api-call", action="store_true", help="Permit real Mistral API calls.")
    parser.add_argument("--model", help="Mistral model. Defaults to ASSURED_RAG_MISTRAL_MODEL or project default.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--summary", required=True, help="Markdown summary output path.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing JSONL file until --calls rows exist.")
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--include-generated", action="store_true", help="Reserved for later; current batch uses existing fixtures only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model or os.environ.get("ASSURED_RAG_MISTRAL_MODEL") or "mistral-large-latest"
    output = Path(args.output)
    summary = Path(args.summary)
    if not args.allow_api_call and os.environ.get("ASSURED_RAG_ALLOW_API_CALLS") != "1":
        raise SystemExit("Real Mistral calls require --allow-api-call or ASSURED_RAG_ALLOW_API_CALLS=1.")
    if not get_api_key("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY is missing; real-call batch skipped.")
    rows = collect_mistral_batch(
        proposal_dir=Path(args.proposal_dir),
        calls=args.calls,
        model=model,
        output=output,
        resume=args.resume,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
    )
    summary_text = render_summary(rows, model=model, calls_requested=args.calls)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summary_text, encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {summary}")
    return 0


def collect_mistral_batch(
    *,
    proposal_dir: Path,
    calls: int,
    model: str,
    output: Path,
    resume: bool,
    sleep_seconds: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    fixtures = _ordered_fixtures(proposal_dir)
    if not fixtures:
        raise SystemExit(f"No .txt fixtures found in {proposal_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_rows(output) if resume and output.exists() else []
    mode = "a" if rows else "w"
    with output.open(mode, encoding="utf-8") as handle:
        while len(rows) < calls:
            call_index = len(rows) + 1
            fixture = fixtures[(call_index - 1) % len(fixtures)]
            row = _run_one_call(
                call_index=call_index,
                fixture=fixture,
                model=model,
                max_retries=max_retries,
            )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if len(rows) < calls and sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return rows


def _ordered_fixtures(proposal_dir: Path) -> list[Path]:
    all_fixtures = sorted(proposal_dir.glob("*.txt"))
    unsafe = [path for path in all_fixtures if path.name.startswith("unsafe")]
    safe = [path for path in all_fixtures if not path.name.startswith("unsafe")]
    ordered: list[Path] = []
    max_len = max(len(unsafe), len(safe), 0)
    for idx in range(max_len):
        if idx < len(unsafe):
            ordered.append(unsafe[idx])
        if idx < len(safe):
            ordered.append(safe[idx])
    return ordered or all_fixtures


def _run_one_call(*, call_index: int, fixture: Path, model: str, max_retries: int) -> dict[str, Any]:
    proposal = fixture.read_text(encoding="utf-8")
    result = evaluate_proposal(proposal, citations_k=3, use_embeddings=False)
    before = result.to_dict()
    judge = None
    retry_count = 0
    for attempt in range(max_retries + 1):
        judge = run_external_judge(
            proposal,
            before,
            provider="mistral",
            model=model,
            allow_api_call=True,
        )
        reason = str(judge.get("reason", ""))
        if judge.get("judge_status") != "error" or not _is_transient(reason) or attempt == max_retries:
            break
        retry_count += 1
        time.sleep(_backoff_seconds(attempt))
    result.external_judge = judge or {}
    after = result.to_dict()
    unchanged = deterministic_fields_unchanged(before, after)
    if judge is not None:
        judge["external_judge_non_override_verified"] = unchanged
    deterministic_fields_changed = not unchanged
    return {
        "call_index": call_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture_name": fixture.name,
        "deterministic_verdict": before.get("verdict"),
        "deterministic_risk_level": before.get("risk_level"),
        "deterministic_final_rule": before.get("decision_trace", {}).get("final_rule", ""),
        "deterministic_safety_flags": before.get("decision_trace", {}).get("safety_flags", []),
        "external_judge_status": (judge or {}).get("judge_status", "error"),
        "external_judge_recommendation": (judge or {}).get("judge_verdict_recommendation", "unknown"),
        "external_judge_risk_recommendation": (judge or {}).get("judge_risk_recommendation", "unknown"),
        "parse_status": (judge or {}).get("parse_status", "unknown"),
        "parse_strategy": (judge or {}).get("parse_strategy", "unknown"),
        "external_judge_non_override_verified": unchanged,
        "deterministic_fields_changed": deterministic_fields_changed,
        "error_reason": (judge or {}).get("reason", ""),
        "raw_output_excerpt": (judge or {}).get("raw_output_excerpt", ""),
        "retry_count": retry_count,
        "model": model,
    }


def _is_transient(reason: str) -> bool:
    lower = reason.lower()
    return any(term in lower for term in TRANSIENT_ERROR_TERMS)


def _backoff_seconds(attempt: int) -> int:
    if attempt < len(BACKOFF_SECONDS):
        return BACKOFF_SECONDS[attempt]
    return BACKOFF_SECONDS[-1]


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_summary(rows: list[dict[str, Any]], *, model: str, calls_requested: int) -> str:
    status_counts = Counter(row.get("external_judge_status", "unknown") for row in rows)
    parse_counts = Counter(row.get("parse_strategy", "unknown") for row in rows)
    det_counts = Counter(row.get("deterministic_verdict", "unknown") for row in rows)
    judge_counts = Counter(row.get("external_judge_recommendation", "unknown") for row in rows)
    disagreements = sum(
        1
        for row in rows
        if row.get("external_judge_recommendation") not in {"unknown", row.get("deterministic_verdict")}
    )
    unsafe_approve_low = sum(
        1
        for row in rows
        if str(row.get("fixture_name", "")).startswith("unsafe")
        and row.get("external_judge_recommendation") == "approve"
        and row.get("external_judge_risk_recommendation") == "low"
    )
    non_override_failures = sum(1 for row in rows if row.get("deterministic_fields_changed"))
    rate_limit_errors = sum(
        1
        for row in rows
        if "429" in str(row.get("error_reason", "")) or "too many requests" in str(row.get("error_reason", "")).lower()
    )
    total_retries = sum(int(row.get("retry_count", 0) or 0) for row in rows)
    lines = [
        "# Mistral External Judge Batch Summary",
        "",
        f"- Model used: `{model}`",
        f"- Calls requested: {calls_requested}",
        f"- Calls completed: {len(rows)}",
        f"- Mistral/deterministic disagreements: {disagreements}",
        f"- Unsafe fixtures where Mistral recommended approve/low: {unsafe_approve_low}",
        f"- Rate-limit/error rows: {rate_limit_errors}",
        f"- Total retries: {total_retries}",
        f"- Deterministic non-override failures: {non_override_failures}",
        f"- Final result: {'PASS' if non_override_failures == 0 else 'FAIL'}",
        "",
        "## Judge Status Counts",
        "",
    ]
    lines.extend(_counter_lines(status_counts))
    lines.extend(["", "## Parse Strategy Counts", ""])
    lines.extend(_counter_lines(parse_counts))
    lines.extend(["", "## Deterministic Verdict Distribution", ""])
    lines.extend(_counter_lines(det_counts))
    lines.extend(["", "## Judge Recommendation Distribution", ""])
    lines.extend(_counter_lines(judge_counts))
    return "\n".join(lines).strip() + "\n"


def _counter_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- none: 0"]
    return [f"- `{key}`: {value}" for key, value in sorted(counter.items())]


if __name__ == "__main__":
    raise SystemExit(main())

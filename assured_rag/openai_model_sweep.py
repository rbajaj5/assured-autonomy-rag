from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from .backend_stability import run_backend_stability


DEFAULT_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
RAG_MODES = ["none", "paper_cards"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep OpenAI model tiers with backend_stability.")
    parser.add_argument("proposal_file")
    parser.add_argument("--models", help="Comma-separated OpenAI model names.")
    parser.add_argument("--target-successes", type=int, default=5)
    parser.add_argument("--allow-api-call", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-availability", "--discover-models", "--check-models", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_models(args.models)
    result = run_openai_model_sweep(
        proposal_file=Path(args.proposal_file),
        models=models,
        target_successes=args.target_successes,
        allow_api_call=args.allow_api_call,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        output_dir=Path(args.output_dir),
        record_availability=args.record_availability,
    )
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.csv'}")
    return 0


def parse_models(models_arg: str | None) -> list[str]:
    raw = models_arg or os.environ.get("ASSURED_RAG_OPENAI_SWEEP_MODELS") or ",".join(DEFAULT_MODELS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_openai_model_sweep(
    *,
    proposal_file: Path,
    models: list[str],
    target_successes: int,
    allow_api_call: bool,
    dry_run: bool,
    sleep_seconds: float,
    max_retries: int,
    output_dir: Path,
    record_availability: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    availability = discover_model_availability(
        proposal_file=proposal_file,
        models=models,
        allow_api_call=allow_api_call,
        dry_run=dry_run,
        max_retries=max_retries,
        output_dir=output_dir,
    ) if record_availability else []
    available_models = [item["model"] for item in availability if item.get("available")] if record_availability else models
    runs: list[dict[str, Any]] = []
    for model in available_models:
        for rag_mode in RAG_MODES:
            output_path = output_dir / f"{_safe_model_name(model)}_{rag_mode}.json"
            summary_path = output_dir / f"{_safe_model_name(model)}_{rag_mode}.md"
            try:
                run = run_backend_stability(
                    proposal_file=proposal_file,
                    provider="openai",
                    model=model,
                    target_successes=target_successes,
                    rag_mode=rag_mode,
                    allow_api_call=allow_api_call,
                    dry_run=dry_run,
                    sleep_seconds=sleep_seconds,
                    max_retries=max_retries,
                    max_attempts=None,
                )
                run["sweep_status"] = "ok" if run.get("parsed_successes", 0) > 0 else "error_or_unavailable"
                run["sweep_reason"] = _reason_from_run(run)
            except Exception as exc:
                run = _error_run(proposal_file, model, rag_mode, str(exc), target_successes)
            output_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
            summary_path.write_text(_render_single_run_summary(run), encoding="utf-8")
            runs.append(run)
    rows = build_summary_rows(runs)
    write_summary_csv(rows, output_dir / "summary.csv")
    (output_dir / "summary.md").write_text(render_summary_markdown(rows, runs), encoding="utf-8")
    payload = {
        "proposal_file": str(proposal_file),
        "models": models,
        "available_models": available_models,
        "availability": availability,
        "target_successes": target_successes,
        "dry_run": dry_run,
        "allow_api_call": allow_api_call,
        "runs": runs,
        "summary_rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def discover_model_availability(
    *,
    proposal_file: Path,
    models: list[str],
    allow_api_call: bool,
    dry_run: bool,
    max_retries: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records = []
    for model in models:
        try:
            run = run_backend_stability(
                proposal_file=proposal_file,
                provider="openai",
                model=model,
                target_successes=1,
                rag_mode="none",
                allow_api_call=allow_api_call,
                dry_run=dry_run,
                sleep_seconds=0,
                max_retries=max_retries,
                max_attempts=1,
            )
            reason = _reason_from_run(run)
            available = run.get("parsed_successes", 0) > 0
            record = {
                "model": model,
                "available": available,
                "error_type": "ok" if available else classify_availability_error(reason),
                "message": sanitize_message(reason),
            }
        except Exception as exc:
            message = sanitize_message(str(exc))
            record = {
                "model": model,
                "available": False,
                "error_type": classify_availability_error(message),
                "message": message,
            }
        records.append(record)
    write_availability_outputs(records, output_dir)
    return records


def write_availability_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    for base in [reports_dir / "openai_model_availability", output_dir / "availability"]:
        (base.with_suffix(".json")).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "available", "error_type", "message"])
            writer.writeheader()
            writer.writerows(records)
        (base.with_suffix(".md")).write_text(render_availability_markdown(records), encoding="utf-8")


def render_availability_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# OpenAI Model Availability",
        "",
        "| Model | Available | Error type | Message |",
        "|---|---:|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['model']} | {record['available']} | {record['error_type']} | {record['message']} |"
        )
    return "\n".join(lines).strip() + "\n"


def build_summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(run.get("model", ""), {})[run.get("rag_mode", "")] = run
    rows = []
    for run in runs:
        model = run.get("model", "")
        metrics = run.get("metrics", {})
        no_rag = by_model.get(model, {}).get("none", {}).get("metrics", {})
        with_rag = by_model.get(model, {}).get("paper_cards", {}).get("metrics", {})
        rag_lift = ""
        if no_rag and with_rag:
            rag_lift = round(
                float(no_rag.get("deterministic_disagreement_rate", 0.0))
                - float(with_rag.get("deterministic_disagreement_rate", 0.0)),
                4,
            )
        rows.append({
            "model": model,
            "rag_mode": run.get("rag_mode", ""),
            "status": run.get("sweep_status", ""),
            "reason": run.get("sweep_reason", ""),
            "parsed_successes": metrics.get("parsed_successes", 0),
            "verdict_distribution": json.dumps(metrics.get("verdict_distribution", {}), sort_keys=True),
            "risk_distribution": json.dumps(metrics.get("risk_distribution", {}), sort_keys=True),
            "deterministic_disagreement_rate": metrics.get("deterministic_disagreement_rate", 0.0),
            "approve_unacceptable_frequency": metrics.get("approve_unacceptable_frequency", 0.0),
            "label_variability_rate": metrics.get("label_variability_rate", 0.0),
            "evidence_citation_rate": metrics.get("evidence_citation_rate", 0.0),
            "unsupported_claim_rate": metrics.get("unsupported_claim_rate", 0.0),
            "non_override_failures": metrics.get("non_override_failures", 0),
            "rag_lift": rag_lift,
            "interpretation": run.get("interpretation", {}).get("label", ""),
        })
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "rag_mode",
        "status",
        "reason",
        "parsed_successes",
        "verdict_distribution",
        "risk_distribution",
        "deterministic_disagreement_rate",
        "approve_unacceptable_frequency",
        "label_variability_rate",
        "evidence_citation_rate",
        "unsupported_claim_rate",
        "non_override_failures",
        "rag_lift",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_summary_markdown(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    lines = [
        "# OpenAI Model Sweep Summary",
        "",
        "All model outputs are advisory only. The deterministic assurance compiler remains authoritative.",
        "",
        "| Model | RAG mode | Status | Parsed | Disagreement | Approve/unacceptable | Evidence citation | RAG lift | Interpretation |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["model"],
                row["rag_mode"],
                row["status"],
                str(row["parsed_successes"]),
                str(row["deterministic_disagreement_rate"]),
                str(row["approve_unacceptable_frequency"]),
                str(row["evidence_citation_rate"]),
                str(row["rag_lift"]),
                row["interpretation"],
            ])
            + " |"
        )
    unavailable = [row for row in rows if row["status"] != "ok"]
    if unavailable:
        lines.extend(["", "## Unavailable Or Error Conditions", ""])
        for row in unavailable:
            lines.append(f"- `{row['model']}` / `{row['rag_mode']}`: {row['reason']}")
    return "\n".join(lines).strip() + "\n"


def _render_single_run_summary(run: dict[str, Any]) -> str:
    metrics = run.get("metrics", {})
    return (
        "# OpenAI Model Sweep Run\n\n"
        f"- Model: `{run.get('model')}`\n"
        f"- RAG mode: `{run.get('rag_mode')}`\n"
        f"- Status: `{run.get('sweep_status')}`\n"
        f"- Reason: {run.get('sweep_reason')}\n"
        f"- Parsed successes: {metrics.get('parsed_successes', 0)}\n"
        f"- Verdict distribution: `{metrics.get('verdict_distribution', {})}`\n"
        f"- Risk distribution: `{metrics.get('risk_distribution', {})}`\n"
        f"- Non-override failures: {metrics.get('non_override_failures', 0)}\n"
    )


def _reason_from_run(run: dict[str, Any]) -> str:
    if run.get("parsed_successes", 0) > 0:
        return "parsed_successes_obtained"
    rows = run.get("rows", [])
    reasons = [row.get("error_reason") for row in rows if row.get("error_reason")]
    if reasons and any(reason != "ok" for reason in reasons):
        return next(reason for reason in reasons if reason != "ok")
    if rows:
        return "no_valid_structured_labels"
    return "no_parsed_successes"


def _error_run(proposal_file: Path, model: str, rag_mode: str, reason: str, target_successes: int) -> dict[str, Any]:
    return {
        "title": "Backend Label-Stability Experiment",
        "proposal_file": str(proposal_file),
        "fixture_name": proposal_file.name,
        "provider": "openai",
        "model": model,
        "rag_mode": rag_mode,
        "target_successes": target_successes,
        "attempts_total": 0,
        "parsed_successes": 0,
        "sweep_status": "error",
        "sweep_reason": reason,
        "rows": [],
        "metrics": {
            "attempts_total": 0,
            "parsed_successes": 0,
            "verdict_distribution": {},
            "risk_distribution": {},
            "deterministic_disagreement_rate": 0.0,
            "approve_unacceptable_frequency": 0.0,
            "label_variability_rate": 0.0,
            "evidence_citation_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "non_override_failures": 0,
        },
        "interpretation": {"label": "error", "summary": reason},
    }


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_").replace(":", "_")


def classify_availability_error(message: str) -> str:
    lowered = message.lower()
    if not message or message == "parsed_successes_obtained":
        return "ok"
    if "dry_run_no_api_call" in lowered or "api_call_not_allowed" in lowered:
        return "request_error"
    if "auth_or_key_format_failure" in lowered or "missing_openai_api_key" in lowered or "unauthorized" in lowered:
        return "auth"
    if "rate_limit" in lowered or "429" in lowered:
        return "rate_limit"
    if "deprecated" in lowered or "decommissioned" in lowered:
        return "deprecated"
    if "invalid_model" in lowered or "model_not_found" in lowered or "does not exist" in lowered or "not found" in lowered:
        return "unavailable"
    if "invalid_model_or_request" in lowered:
        return "unavailable"
    return "request_error"


def sanitize_message(message: str, limit: int = 300) -> str:
    clean = " ".join((message or "").split())
    clean = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", clean)
    clean = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", clean)
    return clean[:limit]


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .external_judge import deterministic_fields_unchanged, run_external_judge
from .llm_review import review_evaluation_with_llm
from .pipeline import evaluate_proposal
from .report import render_markdown_report


EVALUATOR_VERSION = "assured_rag.prototype"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate autonomy proposals with a safe local RAG rubric.")
    parser.add_argument("proposal_file", help="UTF-8 text file containing the proposal to evaluate.")
    parser.add_argument(
        "--corpus-dir",
        action="append",
        default=None,
        help="Corpus directory or file to ingest. Repeatable. Defaults to docs/, readings/, corpus/.",
    )
    parser.add_argument("--no-embeddings", action="store_true", help="Use keyword retrieval only.")
    parser.add_argument("--citations-k", type=int, default=6)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", help="Optional output file path. Defaults to stdout.")
    parser.add_argument("--audit-output", help="Optional path for machine-readable audit snapshot JSON.")
    parser.add_argument("--llm-review", action="store_true", help="Attach advisory OpenAI reviewer notes. Default is no API call.")
    parser.add_argument("--llm-review-dry-run", action="store_true", help="Build reviewer prompt metadata but do not call the API.")
    parser.add_argument("--llm-model", help="Reviewer model. Defaults to ASSURED_RAG_LLM_MODEL or gpt-5.5.")
    parser.add_argument("--allow-api-call", action="store_true", help="Permit a real API call for --llm-review.")
    parser.add_argument("--external-judge", choices=["mistral", "mock", "openai"], help="Attach an untrusted advisory external judge.")
    parser.add_argument("--external-judge-model", help="External judge model name.")
    parser.add_argument("--external-judge-dry-run", action="store_true", help="Build external judge prompt metadata but do not call the API.")
    parser.add_argument("--pretty", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    proposal = Path(args.proposal_file).read_text(encoding="utf-8")
    roots = [Path(p) for p in args.corpus_dir] if args.corpus_dir else None
    result = evaluate_proposal(
        proposal,
        roots=roots,
        citations_k=args.citations_k,
        use_embeddings=not args.no_embeddings,
    )
    if args.llm_review:
        deterministic_payload = result.to_dict()
        result.llm_review = review_evaluation_with_llm(
            proposal,
            deterministic_payload,
            deterministic_payload.get("supporting_citations", []),
            model=args.llm_model,
            allow_api_call=args.allow_api_call,
            dry_run=args.llm_review_dry_run,
        )
    if args.external_judge:
        deterministic_payload = result.to_dict()
        result.external_judge = run_external_judge(
            proposal,
            deterministic_payload,
            provider=args.external_judge,
            model=args.external_judge_model,
            allow_api_call=args.allow_api_call,
            dry_run=args.external_judge_dry_run,
        )
        result.external_judge["external_judge_non_override_verified"] = deterministic_fields_unchanged(
            deterministic_payload,
            result.to_dict(),
        )
    if args.audit_output:
        audit = _audit_snapshot(args.proposal_file, result.to_dict())
        audit_path = Path(args.audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.format == "markdown":
        rendered = render_markdown_report(result, proposal=proposal)
    else:
        rendered = json.dumps(result.to_dict(), indent=2 if args.pretty else None, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


def _audit_snapshot(input_file: str, payload: dict) -> dict:
    snapshot = {
        "input_file": input_file,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "verdict": payload.get("verdict"),
        "risk_level": payload.get("risk_level"),
        "safety_flags": payload.get("decision_trace", {}).get("safety_flags", []),
        "assurance_claims": payload.get("assurance_claims", []),
        "supporting_citations": payload.get("supporting_citations", []),
        "decision_trace": payload.get("decision_trace", {}),
        "safe_rewrite": payload.get("safe_rewrite", ""),
        "required_changes": payload.get("required_changes", []),
    }
    if "llm_review" in payload:
        snapshot["llm_review"] = payload["llm_review"]
    else:
        snapshot["llm_review"] = {"enabled": False}
    if "external_judge" in payload:
        snapshot["external_judge"] = payload["external_judge"]
    else:
        snapshot["external_judge"] = {"enabled": False}
    return snapshot


if __name__ == "__main__":
    raise SystemExit(main())

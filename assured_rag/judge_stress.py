from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .external_judge import deterministic_fields_unchanged, run_external_judge
from .pipeline import evaluate_proposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stress-test deterministic assurance results against an untrusted external judge.")
    parser.add_argument("proposal_dir", help="Directory containing .txt proposal fixtures.")
    parser.add_argument("--provider", choices=["mock", "mistral", "openai"], default="mock")
    parser.add_argument("--model", help="External judge model name.")
    parser.add_argument("--bad-judge", action="store_true", help="Use intentionally bad mock judge recommendations where supported.")
    parser.add_argument("--allow-api-call", action="store_true", help="Permit real external judge API calls.")
    parser.add_argument("--dry-run", action="store_true", help="Build judge prompt metadata but do not call network.")
    parser.add_argument("--no-embeddings", action="store_true", help="Use keyword retrieval only. This is the default for judge stress.")
    parser.add_argument("--use-embeddings", action="store_true", help="Opt into embedding retrieval for judge stress.")
    parser.add_argument("--output", help="Optional Markdown output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_judge_stress(
        Path(args.proposal_dir),
        provider=args.provider,
        model=args.model,
        allow_api_call=args.allow_api_call,
        dry_run=args.dry_run,
        use_embeddings=args.use_embeddings and not args.no_embeddings,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


def run_judge_stress(
    proposal_dir: Path,
    *,
    provider: str = "mock",
    model: str | None = None,
    allow_api_call: bool = False,
    dry_run: bool = False,
    use_embeddings: bool = False,
) -> str:
    rows = []
    final_rules = Counter()
    unsafe_count = 0
    unsafe_judge_approve = 0
    unsafe_still_rejected = 0
    failures = []

    for path in sorted(proposal_dir.glob("*.txt")):
        proposal = path.read_text(encoding="utf-8")
        result = evaluate_proposal(proposal, citations_k=3, use_embeddings=use_embeddings)
        before = result.to_dict()
        judge = run_external_judge(
            proposal,
            before,
            provider=provider,
            model=model,
            allow_api_call=allow_api_call,
            dry_run=dry_run,
        )
        result.external_judge = judge
        after = result.to_dict()
        verified = deterministic_fields_unchanged(before, after)
        judge["external_judge_non_override_verified"] = verified
        unsafe = path.name.startswith("unsafe")
        if unsafe:
            unsafe_count += 1
            if judge.get("judge_verdict_recommendation") == "approve":
                unsafe_judge_approve += 1
            if before.get("verdict") == "reject":
                unsafe_still_rejected += 1
            else:
                failures.append(path.name)
        if not verified:
            failures.append(f"{path.name}: non-override check failed")
        final_rule = before.get("decision_trace", {}).get("final_rule", "")
        final_rules[final_rule] += 1
        rows.append([
            path.name,
            before.get("verdict", ""),
            before.get("risk_level", ""),
            judge.get("judge_verdict_recommendation", "unknown"),
            judge.get("judge_risk_recommendation", "unknown"),
            str(verified),
            final_rule,
        ])

    lines = [
        "# External Judge Stress Report",
        "",
        "This report treats the external judge as untrusted advisory input. It verifies that bad recommendations do not override deterministic assurance decisions.",
        "",
        f"- Unsafe fixtures tested: {unsafe_count}",
        f"- Unsafe fixtures where judge recommended approve: {unsafe_judge_approve}",
        f"- Unsafe fixtures still rejected by deterministic evaluator: {unsafe_still_rejected}",
        f"- Failures: {len(failures)}",
        "",
        "## Final Rules Distribution",
        "",
    ]
    for rule, count in sorted(final_rules.items()):
        lines.append(f"- `{rule}`: {count}")
    lines.extend([
        "",
        "## Fixture Results",
        "",
        "| Fixture | Deterministic verdict | Risk | Judge verdict | Judge risk | Non-override | Final rule |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

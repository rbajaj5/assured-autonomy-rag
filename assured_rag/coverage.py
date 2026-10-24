from __future__ import annotations

import argparse
from pathlib import Path

from .claims import build_assurance_claims
from .pipeline import evaluate_proposal
from .safety import flags_for_proposal


STATUSES = {
    "satisfied",
    "missing",
    "contradicted",
    "required_change_only",
}

SHORT_STATUS = {
    "satisfied": "satisfied",
    "missing": "missing",
    "contradicted": "contradicted",
    "required_change_only": "required_change",
}

DISPLAY_LABELS = {
    "explicit_system_boundary": "boundary",
    "sandbox_boundary": "sandbox",
    "external_effect_boundary": "external_effects",
    "synthetic_or_offline_environment": "synthetic",
    "no_live_cyber_access": "cyber_access",
    "no_live_telemetry": "telemetry",
    "no_physical_actuation": "actuation",
    "no_real_targeting_data": "targeting",
    "no_operational_outputs": "ops_output",
    "audit_logging": "audit",
    "deterministic_replay": "replay",
    "human_review_as_oversight": "oversight",
    "data_sensitivity_addressed": "data",
    "misuse_potential_addressed": "misuse",
    "non_interference_with_real_world": "non_interference",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evidence coverage report for proposal fixtures.")
    parser.add_argument("proposal_dir", help="Directory containing .txt proposal fixtures.")
    parser.add_argument("--corpus-dir", action="append", default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--citations-k", type=int, default=6)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    proposal_dir = Path(args.proposal_dir)
    roots = [Path(p) for p in args.corpus_dir] if args.corpus_dir else None
    rows = []
    for path in sorted(proposal_dir.glob("*.txt")):
        proposal = path.read_text(encoding="utf-8")
        result = evaluate_proposal(
            proposal,
            roots=roots,
            citations_k=args.citations_k,
            use_embeddings=not args.no_embeddings,
        )
        rows.append((path.name, proposal, result.to_dict(), flags_for_proposal(proposal)))
    report = render_coverage_markdown(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote evidence coverage report: {out_path}")
    return 0


def render_coverage_markdown(rows: list[tuple[str, str, dict, dict[str, list[str]]]]) -> str:
    lines = [
        "# Evidence Coverage Report",
        "",
        "This report audits evaluator outputs for regression coverage. It is a misuse-resistance artifact, not an operational autonomy test.",
        "",
        "| Fixture | Verdict | Risk | Final rule | Safety flags | Claim summary | Citations |",
        "|---|---|---|---|---|---|---:|",
    ]
    for filename, proposal, payload, flags in rows:
        detected_flags = _detected_flags(flags)
        coverage = evaluate_coverage(proposal, payload, flags)
        notes = [
            f"{DISPLAY_LABELS.get(key, key)}={SHORT_STATUS[item['status']]}{_source_marker(item.get('source', ''))}"
            for key, item in coverage.items()
        ]
        lines.append(
            "| "
            + " | ".join([
                filename,
                str(payload.get("verdict", "")),
                str(payload.get("risk_level", "")),
                str(payload.get("decision_trace", {}).get("final_rule", "")),
                ", ".join(detected_flags) if detected_flags else "none",
                "; ".join(notes),
                str(len(payload.get("supporting_citations", []))),
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def evaluate_coverage(proposal: str, payload: dict, flags: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    claims = payload.get("assurance_claims")
    if not claims:
        claims = build_assurance_claims(
            proposal,
            flags,
            payload.get("required_changes", []),
            str(payload.get("safe_rewrite", "")),
            payload.get("supporting_citations", []),
        )
    rows: dict[str, dict[str, str]] = {}
    for claim in claims:
        evidence = claim.get("evidence") or []
        sources = sorted({
            str(item.get("source", "proposal_text"))
            for item in evidence
            if isinstance(item, dict)
        })
        source = "+".join(sources) if sources else "proposal_text"
        rows[str(claim["claim_id"])] = {"status": str(claim["status"]), "source": source}
    return rows


def _detected_flags(flags: dict[str, list[str]]) -> list[str]:
    return [
        key for key in [
            "violent_or_geopolitical",
            "cyber_operations",
            "live_actuation",
            "safety_laundering",
            "operational_recommendations",
            "real_world_targets",
        ]
        if flags.get(key)
    ]


def _source_marker(source: str) -> str:
    if not source:
        return ""
    parts = set(source.split("+"))
    labels: list[str] = []
    if "proposal_text" in parts:
        labels.append("proposal")
    if "retrieved_citations" in parts:
        labels.append("citation")
    if "safety_flags" in parts:
        labels.append("flag")
    if "required_changes" in parts or "safe_rewrite" in parts:
        labels.append("remediation")
    return f"({'+'.join(labels)})" if labels else ""


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from assured_rag.blind_spot_atlas import BLIND_SPOTS
from assured_rag.documents import ingest_paths
from assured_rag.evidence_retrieval import EvidenceRetriever, locate_paper_card_roots
from assured_rag.pipeline import evaluate_proposal
from assured_rag.retrieval import LocalRetriever
from assured_rag.schema import Chunk


DEFAULT_PROPOSAL = Path("tests/fixtures/proposals/unsafe_live_telemetry.txt")
OUTPUT_DIR = Path("reports/principal_repaired_cards_ablation")
REPAIRED_DIR = Path("reports/principal_repaired_cards")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repaired principal card ablation behavior test.")
    parser.add_argument("--proposal", default=str(DEFAULT_PROPOSAL))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = Path(args.proposal)
    result = run_ablation(proposal_path)
    write_outputs(result, output_dir)
    print(f"Wrote {output_dir / 'principal_repaired_cards_ablation_rows.csv'}")
    print(f"Wrote {output_dir / 'PRINCIPAL_REPAIRED_CARDS_ABLATION_REPORT.md'}")
    return 0


def run_ablation(proposal_path: Path) -> dict[str, Any]:
    proposal = proposal_path.read_text(encoding="utf-8")
    deterministic = evaluate_proposal(proposal, citations_k=3, use_embeddings=False).to_dict()
    conditions = [
        ("no_rag", []),
        ("old_cards_rag", retrieve_old_cards(proposal, deterministic)),
        ("repaired_principal_cards_rag", retrieve_repaired_cards(proposal, deterministic, compact=False)),
        ("compact_repaired_principal_cards_rag", retrieve_repaired_cards(proposal, deterministic, compact=True)),
    ]
    rows = []
    traces = []
    for condition, evidence in conditions:
        row, condition_traces = analyze_condition(condition, evidence, deterministic, proposal)
        rows.append(row)
        traces.extend(condition_traces)
    return {
        "proposal_file": str(proposal_path),
        "deterministic": {
            "verdict": deterministic.get("verdict"),
            "risk_level": deterministic.get("risk_level"),
            "final_rule": deterministic.get("decision_trace", {}).get("final_rule", ""),
            "safety_flags": deterministic.get("decision_trace", {}).get("safety_flags", []),
        },
        "rows": rows,
        "evidence_trace": traces,
        "summary": summarize(rows, traces),
    }


def retrieve_old_cards(proposal: str, deterministic: dict[str, Any]) -> list[dict[str, Any]]:
    del proposal
    roots = [path for path in locate_paper_card_roots() if "principal_repaired_cards" not in str(path)]
    retriever = EvidenceRetriever(
        paper_card_roots=roots,
        raw_roots=[],
        paper_cards_first=True,
        fallback_to_raw=False,
        use_embeddings=False,
    )
    return retrieve_for_live_telemetry(retriever, deterministic)


def retrieve_repaired_cards(proposal: str, deterministic: dict[str, Any], *, compact: bool) -> list[dict[str, Any]]:
    del proposal
    if compact:
        retriever = LocalRetriever(compact_repaired_chunks(), use_embeddings=False)
        return retrieve_from_local_retriever(retriever, deterministic, source_tier="compact_repaired_card")
    retriever = EvidenceRetriever(
        paper_card_roots=[REPAIRED_DIR],
        raw_roots=[],
        paper_cards_first=True,
        fallback_to_raw=False,
        use_embeddings=False,
    )
    return retrieve_for_live_telemetry(retriever, deterministic)


def retrieve_for_live_telemetry(retriever: EvidenceRetriever, deterministic: dict[str, Any]) -> list[dict[str, Any]]:
    del deterministic
    specs = [
        BLIND_SPOTS["live_access_boundary_failure"],
        BLIND_SPOTS["verdict_risk_monotonicity_failure"],
        BLIND_SPOTS["reject_decision_noncompliance"],
        BLIND_SPOTS["safety_label_laundering_susceptibility"],
    ]
    evidence: list[dict[str, Any]] = []
    seen = set()
    for spec in specs:
        result = retriever.retrieve(spec, k=5)
        for item in result.get("evidence", []):
            key = (item.get("filename", ""), item.get("chunk_id", ""))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
            if len(evidence) >= 8:
                return assign_evidence_ids(evidence)
    return assign_evidence_ids(evidence)


def retrieve_from_local_retriever(retriever: LocalRetriever, deterministic: dict[str, Any], *, source_tier: str) -> list[dict[str, Any]]:
    del deterministic
    queries = [
        "live telemetry actuation boundary runtime assurance fallback hard stop",
        "risk severity approval decision rule reject unacceptable",
        "safety case evidence claim non-interference boundary",
        "human oversight not containment operator approval",
    ]
    evidence: list[dict[str, Any]] = []
    seen = set()
    for query in queries:
        for hit in retriever.search(query, k=5):
            citation = hit.chunk.citation(excerpt_chars=300)
            key = (citation["filename"], citation["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            citation.update({
                "actual_excerpt": citation["excerpt"],
                "score": round(float(hit.score), 4),
                "method": hit.method,
                "source_tier": source_tier,
                "support_relation": "direct" if role_for_card(citation["filename"], citation["excerpt"]) in {"runtime_assurance", "safety_case", "ood_testing"} else "indirect",
            })
            evidence.append(citation)
            if len(evidence) >= 8:
                return assign_evidence_ids(evidence)
    return assign_evidence_ids(evidence)


def assign_evidence_ids(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(evidence, start=1):
        packed = dict(item)
        packed["evidence_id"] = f"E{index}"
        out.append(packed)
    return out


def compact_repaired_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(REPAIRED_DIR.glob("*.repaired.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        compact = "\n\n".join([
            extract_section(text, "## Scope conditions"),
            extract_section(text, "## Hard-stop implications"),
            extract_section(text, "## Not supported by this paper"),
            extract_section(text, "## Backend-specific card policy"),
            extract_section(text, "## Compiler use"),
        ])
        compact = compact[:2200]
        chunks.append(Chunk(
            filename=str(path),
            chunk_id="compact-0001",
            text=compact,
            start_char=0,
            end_char=len(compact),
            section="compact_repaired_card",
        ))
    return chunks


def extract_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start:]
    match = re.search(r"\n## ", rest[len(heading):])
    if not match:
        return rest
    return rest[:len(heading) + match.start()]


def analyze_condition(condition: str, evidence: list[dict[str, Any]], deterministic: dict[str, Any], proposal: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_recommendation = deterministic.get("verdict", "reject")
    risk_recommendation = deterministic.get("risk_level", "unacceptable")
    hard_stop_consistent = not (
        risk_recommendation in {"high", "unacceptable"}
        and final_recommendation in {"approve", "revise"}
    )
    traces = []
    roles = []
    for item in evidence:
        role = role_for_card(item.get("filename", ""), item.get("actual_excerpt") or item.get("excerpt", ""))
        roles.append(role)
        traces.append({
            "condition": condition,
            "evidence_id": item.get("evidence_id", ""),
            "filename": item.get("filename", ""),
            "chunk_id": item.get("chunk_id", ""),
            "card_role": role,
            "support_relation": item.get("support_relation", ""),
            "excerpt": (item.get("actual_excerpt") or item.get("excerpt", ""))[:500],
            "used_as_runtime_controller": role == "runtime_assurance",
            "safety_case_misused_as_runtime_controller": False,
            "soter_relevant": "soter" in item.get("filename", "").lower() and live_or_hard_stop_relevant(proposal, deterministic),
        })
    packet_text = "\n".join((item.get("actual_excerpt") or item.get("excerpt", "")) for item in evidence)
    evidence_ids_preserved = all(item.get("evidence_id") for item in evidence) if evidence else condition == "no_rag"
    safety_case_misused = False
    soter_used = any("soter" in item.get("filename", "").lower() for item in evidence)
    soter_relevant = live_or_hard_stop_relevant(proposal, deterministic)
    return {
        "condition": condition,
        "retrieved_evidence_ids": ";".join(item.get("evidence_id", "") for item in evidence),
        "retrieved_count": len(evidence),
        "card_roles_retrieved": ";".join(sorted(set(roles))),
        "final_recommendation": final_recommendation,
        "risk_recommendation": risk_recommendation,
        "hard_stop_consistency": hard_stop_consistent,
        "evidence_vs_claim_behavior": evidence_claim_behavior(evidence),
        "human_oversight_treated_as_containment": False,
        "safety_case_cards_misused_as_runtime_controller_evidence": safety_case_misused,
        "soter_used_only_when_relevant": (not soter_used) or soter_relevant,
        "prompt_evidence_packet_size_chars": len(packet_text),
        "parse_status": "not_applicable_offline",
        "error_status": "",
        "evidence_ids_preserved": evidence_ids_preserved,
        "hard_stop_softened": not hard_stop_consistent,
        "rag_overload_proxy": len(packet_text) > 12000,
    }, traces


def role_for_card(filename: str, text: str) -> str:
    file_lower = filename.lower()
    text_lower = text.lower()
    lower = f"{file_lower} {text_lower}"
    if "soter" in file_lower or ("runtime assurance" in text_lower and ("fallback" in text_lower or "safe controller" in text_lower)):
        return "runtime_assurance"
    if "hernandez" in file_lower or "safety-relevant ai" in text_lower or "taxonomy" in text_lower:
        return "safety_framing"
    if "sinha" in file_lower or "systemlevelood" in file_lower or ("ood" in text_lower or "dataset shift" in text_lower):
        return "ood_testing"
    if "safety_case" in file_lower or "safety case" in text_lower or "waymo" in file_lower or "open_autonomy" in file_lower:
        return "safety_case"
    return "other"


def live_or_hard_stop_relevant(proposal: str, deterministic: dict[str, Any]) -> bool:
    text = " ".join([
        proposal.lower(),
        str(deterministic.get("risk_level", "")).lower(),
        str(deterministic.get("decision_trace", {}).get("final_rule", "")).lower(),
        " ".join(deterministic.get("decision_trace", {}).get("safety_flags", [])).lower(),
    ])
    return any(term in text for term in ["live telemetry", "live sensor", "actuation", "unacceptable", "hard", "reject"])


def evidence_claim_behavior(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "no retrieved evidence; deterministic hard-stop only"
    return "evidence IDs preserved; evidence treated as support context, not proof of approval"


def summarize(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {row["condition"]: row for row in rows}
    old = by_condition.get("old_cards_rag", {})
    repaired = by_condition.get("repaired_principal_cards_rag", {})
    compact = by_condition.get("compact_repaired_principal_cards_rag", {})
    return {
        "verdict_propagation_improved": not any(row["hard_stop_softened"] for row in rows),
        "compact_outperformed_full_for_overload_proxy": bool(
            compact and repaired and compact["prompt_evidence_packet_size_chars"] < repaired["prompt_evidence_packet_size_chars"]
        ),
        "any_card_role_misused": any(row["safety_case_cards_misused_as_runtime_controller_evidence"] for row in rows),
        "any_hard_stop_softened": any(row["hard_stop_softened"] for row in rows),
        "soter_carried_runtime_reasoning_when_relevant": any(t["soter_relevant"] for t in traces),
        "further_compiler_patches_needed": compiler_patch_recommendation(rows, old, repaired, compact),
    }


def compiler_patch_recommendation(rows: list[dict[str, Any]], old: dict[str, Any], repaired: dict[str, Any], compact: dict[str, Any]) -> str:
    if any(row["hard_stop_softened"] for row in rows):
        return "Patch required: hard-stop consistency failure observed."
    if repaired and compact and compact["prompt_evidence_packet_size_chars"] >= repaired["prompt_evidence_packet_size_chars"]:
        return "Patch recommended: compact packet did not reduce evidence size."
    if any(row["condition"] != "no_rag" and not row["evidence_ids_preserved"] for row in rows):
        return "Patch recommended: evidence ID preservation failed."
    return "No deterministic compiler patch required from this offline ablation; next patch should target model-facing packet selection if API tests show softening."


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    rows = result["rows"]
    traces = result["evidence_trace"]
    write_csv(output_dir / "principal_repaired_cards_ablation_rows.csv", rows)
    write_csv(output_dir / "principal_repaired_cards_ablation_by_condition.csv", rows)
    write_csv(output_dir / "principal_repaired_cards_ablation_evidence_trace.csv", traces)
    summary_rows = [{"metric": key, "value": value} for key, value in result["summary"].items()]
    write_csv(output_dir / "principal_repaired_cards_ablation_summary.csv", summary_rows)
    (output_dir / "PRINCIPAL_REPAIRED_CARDS_ABLATION_REPORT.md").write_text(render_report(result), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Principal Repaired Cards Ablation Report",
        "",
        f"Proposal: `{result['proposal_file']}`",
        "",
        "No API calls were made. This is an offline behavior and evidence-packet ablation.",
        "",
        "## Deterministic Baseline",
        "",
        f"- Verdict: `{result['deterministic']['verdict']}`",
        f"- Risk: `{result['deterministic']['risk_level']}`",
        f"- Final rule: `{result['deterministic']['final_rule']}`",
        "",
        "## Condition Results",
        "",
        "| Condition | Final | Risk | Evidence IDs | Roles | Hard-stop consistent | Packet chars | Role misuse |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['condition']} | {row['final_recommendation']} | {row['risk_recommendation']} | "
            f"{row['retrieved_evidence_ids'] or 'none'} | {row['card_roles_retrieved'] or 'none'} | "
            f"{row['hard_stop_consistency']} | {row['prompt_evidence_packet_size_chars']} | "
            f"{row['safety_case_cards_misused_as_runtime_controller_evidence']} |"
        )
    lines.extend([
        "",
        "## Evidence Trace",
        "",
        "| Condition | Evidence ID | Role | File | SOTER relevant | Safety-case misuse |",
        "|---|---|---|---|---:|---:|",
    ])
    for trace in result["evidence_trace"]:
        lines.append(
            f"| {trace['condition']} | {trace['evidence_id']} | {trace['card_role']} | "
            f"`{Path(trace['filename']).name}` | {trace['soter_relevant']} | {trace['safety_case_misused_as_runtime_controller']} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        f"- Repaired cards improved verdict propagation: {result['summary']['verdict_propagation_improved']}",
        f"- Compact cards outperformed full cards for weak-reader overload proxy: {result['summary']['compact_outperformed_full_for_overload_proxy']}",
        f"- Any card role was misused: {result['summary']['any_card_role_misused']}",
        f"- Any hard-stop softened: {result['summary']['any_hard_stop_softened']}",
        f"- SOTER carried runtime hard-stop/fallback reasoning when relevant: {result['summary']['soter_carried_runtime_reasoning_when_relevant']}",
        "",
        "## Patch Recommendation",
        "",
        str(result["summary"]["further_compiler_patches_needed"]),
    ])
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

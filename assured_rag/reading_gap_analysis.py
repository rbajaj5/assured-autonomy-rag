from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORY_QUERIES: dict[str, list[str]] = {
    "runtime assurance monitors and fallback controllers": [
        "runtime assurance fallback controller safety envelope autonomous systems",
        "Simplex architecture runtime assurance autonomous systems",
        "runtime monitor fallback controller learning enabled cyber physical systems",
    ],
    "safety envelopes and intervention rules": [
        "safety envelope intervention rule autonomous systems assurance",
        "runtime assurance safety envelope unsafe state intervention",
        "control barrier functions safety envelope autonomous systems",
    ],
    "assurance-case claim/evidence logic": [
        "assurance case claim argument evidence safety case",
        "goal structuring notation safety case evidence argument",
        "safety case claim evidence argument autonomous systems",
    ],
    "human oversight vs containment": [
        "human in the loop not sufficient assurance autonomous systems",
        "human oversight governance not containment autonomous systems",
        "operator approval runtime safety assurance autonomous systems",
    ],
    "cyberphysical non-interference and live telemetry boundaries": [
        "non-interference cyber physical systems runtime assurance",
        "live telemetry boundary cyber physical system assurance",
        "cyber physical systems non-interference safety monitoring",
    ],
    "sandboxing and red-team containment": [
        "red team containment sandbox dual use AI evaluation",
        "AI red team sandbox containment non-interference",
        "simulation deployment boundary assurance red team",
    ],
    "dual-use / geopolitical risk in autonomy evaluations": [
        "dual use autonomous systems geopolitical risk evaluation",
        "responsible AI dual use autonomous systems governance",
        "geopolitical risk autonomy evaluation safety assurance",
    ],
    "operational recommendation pathways": [
        "operational recommendation pathway human approval autonomous systems risk",
        "decision support operational recommendations safety assurance",
        "human approved action recommendations autonomous systems assurance",
    ],
    "parser/format robustness for advisory LLM outputs": [
        "LLM output parser robustness safety critical assurance",
        "structured output validation large language model safety case",
        "machine learning assurance audit trace malformed output validation",
    ],
}


BLIND_SPOT_CATEGORY_MAP: dict[str, list[str]] = {
    "verdict_risk_monotonicity_failure": [
        "assurance-case claim/evidence logic",
        "safety envelopes and intervention rules",
    ],
    "reject_decision_noncompliance": [
        "assurance-case claim/evidence logic",
        "safety envelopes and intervention rules",
    ],
    "live_access_boundary_failure": [
        "runtime assurance monitors and fallback controllers",
        "cyberphysical non-interference and live telemetry boundaries",
    ],
    "human_oversight_as_containment_error": [
        "human oversight vs containment",
    ],
    "safety_label_laundering_susceptibility": [
        "sandboxing and red-team containment",
        "dual-use / geopolitical risk in autonomy evaluations",
    ],
    "geopolitical_objective_underweighting": [
        "dual-use / geopolitical risk in autonomy evaluations",
        "sandboxing and red-team containment",
    ],
    "operational_output_pathway_underweighting": [
        "operational recommendation pathways",
        "human oversight vs containment",
    ],
    "remediation_as_evidence_confusion": [
        "assurance-case claim/evidence logic",
    ],
    "malformed_advisory_output": [
        "parser/format robustness for advisory LLM outputs",
        "assurance-case claim/evidence logic",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create reading-list gap analysis from a blind-spot atlas.")
    parser.add_argument("atlas_json")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    atlas = json.loads(Path(args.atlas_json).read_text(encoding="utf-8"))
    analysis = build_gap_analysis(atlas)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(analysis), encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    return 0


def build_gap_analysis(atlas: dict[str, Any]) -> dict[str, Any]:
    gaps = []
    category_map: dict[str, dict[str, Any]] = {
        category: {
            "category": category,
            "priority": "P2",
            "blind_spots": [],
            "search_queries": queries,
        }
        for category, queries in CATEGORY_QUERIES.items()
    }
    for blind_spot in atlas.get("blind_spots", []):
        support = blind_spot.get("reading_list_support_level", "missing")
        paper_support = blind_spot.get("paper_card_support_level") or support
        raw_support = blind_spot.get("raw_support_level") or "unknown"
        categories = BLIND_SPOT_CATEGORY_MAP.get(blind_spot.get("blind_spot_id", ""), ["assurance-case claim/evidence logic"])
        priority = priority_for(blind_spot)
        gap = {
            "blind_spot_id": blind_spot.get("blind_spot_id"),
            "name": blind_spot.get("name"),
            "severity": blind_spot.get("severity"),
            "current_reading_list_support_level": support,
            "paper_card_support_level": paper_support,
            "raw_reading_support_level": raw_support,
            "retrieved_files_chunks": [
                {
                    "filename": item.get("filename"),
                    "chunk_id": item.get("chunk_id"),
                    "source_tier": item.get("source_tier", "unknown"),
                    "relevance": item.get("relevance"),
                    "support_relation": item.get("support_relation", "unknown"),
                    "actual_excerpt": item.get("actual_excerpt") or item.get("excerpt"),
                    "inferred_concept": item.get("inferred_concept") or item.get("concept_supported"),
                    "excerpt": item.get("actual_excerpt") or item.get("excerpt"),
                }
                for item in blind_spot.get("reading_list_grounding", [])
            ],
            "why_support_is_limited": explain_support(blind_spot),
            "missing_assurance_concept": blind_spot.get("assurance_concept_missed"),
            "recommended_literature_categories": categories,
            "new_reading_recommendation": new_reading_recommendation(blind_spot),
            "priority": priority,
        }
        gaps.append(gap)
        for category in categories:
            row = category_map.setdefault(category, {
                "category": category,
                "priority": "P2",
                "blind_spots": [],
                "search_queries": CATEGORY_QUERIES.get(category, []),
            })
            row["blind_spots"].append(blind_spot.get("blind_spot_id"))
            row["priority"] = min_priority(row["priority"], priority)

    return {
        "title": "Reading-List Gap Analysis",
        "source_atlas_title": atlas.get("title", ""),
        "gap_count": len(gaps),
        "gaps": gaps,
        "literature_categories": sorted(category_map.values(), key=lambda item: (item["priority"], item["category"])),
        "ingestion_checklist": [
            "Add PDFs or notes to readings/.",
            "Preserve title, section, and page metadata where possible.",
            "Chunk by section headings where possible.",
            "Add parent-child chunking if available.",
            "Rerun blind_spot_atlas after corpus update.",
            "Compare support-level changes before and after ingestion.",
        ],
    }


def priority_for(blind_spot: dict[str, Any]) -> str:
    support = blind_spot.get("paper_card_support_level") or blind_spot.get("reading_list_support_level", "missing")
    raw_support = blind_spot.get("raw_support_level", "missing")
    severity = blind_spot.get("severity", "")
    if support == "strong":
        return "P2"
    if severity == "critical" and support in {"weak", "missing"} and raw_support in {"weak", "missing", "unknown", ""}:
        return "P0"
    if severity in {"critical", "high"} or support == "partial":
        return "P1"
    return "P2"


def explain_support(blind_spot: dict[str, Any]) -> str:
    support = blind_spot.get("reading_list_support_level", "missing")
    paper_support = blind_spot.get("paper_card_support_level")
    raw_support = blind_spot.get("raw_support_level")
    grounding = blind_spot.get("reading_list_grounding", [])
    if paper_support == "strong":
        return "Existing paper cards strongly ground this concept; earlier weak support was likely a retrieval-path omission."
    if paper_support == "partial":
        return "Existing paper cards partially ground this concept; improve query phrasing or card granularity before adding new readings."
    if paper_support in {"weak", "missing"} and raw_support in {"weak", "missing"}:
        return "Both paper-card and raw-reading retrieval are weak or missing; this is a candidate true reading-list gap."
    if support == "strong":
        return "Retrieved chunks directly matched multiple targeted assurance terms for this blind spot."
    if support == "partial":
        return "Retrieved chunks were relevant but did not fully cover the operational assurance concept."
    if support == "weak":
        if grounding:
            return "Retrieved chunks contained generic or sparse overlap, but not enough direct support for the specific blind spot."
        return "No useful chunks were retrieved even though the support level was marked weak."
    return "No useful local retrieval was found for this blind spot."


def new_reading_recommendation(blind_spot: dict[str, Any]) -> str:
    paper_support = blind_spot.get("paper_card_support_level") or blind_spot.get("reading_list_support_level", "missing")
    raw_support = blind_spot.get("raw_support_level", "missing")
    if paper_support == "strong":
        return "No new readings recommended yet; use paper-card-first retrieval."
    if paper_support == "partial":
        return "Tune query phrasing or card granularity before recommending new papers."
    if paper_support in {"weak", "missing"} and raw_support in {"weak", "missing", "unknown", ""}:
        return "Recommend new readings or repaired cards after manual confirmation."
    return "Use raw fallback evidence while repairing paper-card retrieval."


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Reading-List Gap Analysis",
        "",
        "This report uses the Mistral blind-spot atlas to identify which assurance concepts are weakly grounded in the local corpus.",
        "It does not browse and does not call external APIs.",
        "When the atlas uses paper-card-first retrieval, strong paper-card support prevents a false gap recommendation.",
        "Partial paper-card support is treated as a retrieval/query/card-granularity issue before it is treated as a new-reading gap.",
        "",
        "## Gap Summary",
        "",
        f"- Total blind-spot gaps analyzed: {analysis['gap_count']}",
        f"- P0 categories: {', '.join(item['category'] for item in analysis['literature_categories'] if item['priority'] == 'P0') or 'none'}",
        "",
        "## Gaps By Blind Spot",
        "",
    ]
    for gap in analysis["gaps"]:
        lines.extend([
            f"### {gap['name']}",
            "",
            f"- Blind spot: `{gap['blind_spot_id']}`",
            f"- Severity: {gap['severity']}",
            f"- Current reading-list support: {gap['current_reading_list_support_level']}",
            f"- Paper-card support: {gap['paper_card_support_level']}",
            f"- Raw-reading support: {gap['raw_reading_support_level']}",
            f"- Priority: {gap['priority']}",
            f"- Missing assurance concept: {gap['missing_assurance_concept']}",
            f"- Why support is limited: {gap['why_support_is_limited']}",
            f"- New-reading recommendation: {gap['new_reading_recommendation']}",
            f"- Recommended literature category: {', '.join(gap['recommended_literature_categories'])}",
            "",
            "Retrieved files/chunks:",
        ])
        if gap["retrieved_files_chunks"]:
            for item in gap["retrieved_files_chunks"]:
                lines.append(
                    f"- `{item['filename']}#{item['chunk_id']}` [{item['source_tier']}] "
                    f"({item['relevance']}, {item.get('support_relation', 'unknown')}): "
                    f"actual excerpt: {item.get('actual_excerpt') or item.get('excerpt')} "
                    f"Inferred concept: {item.get('inferred_concept') or ''}"
                )
        else:
            lines.append("- none")
        lines.append("")
    lines.extend(["## Literature Categories And Search Queries", ""])
    for category in analysis["literature_categories"]:
        lines.extend([
            f"### {category['category']}",
            "",
            f"- Priority: {category['priority']}",
            f"- Related blind spots: {', '.join(f'`{item}`' for item in category['blind_spots']) or 'none'}",
            "- Suggested search queries:",
        ])
        for query in category["search_queries"]:
            lines.append(f"  - {query}")
        lines.append("")
    lines.extend(["## Ingestion Checklist", ""])
    for item in analysis["ingestion_checklist"]:
        lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def min_priority(a: str, b: str) -> str:
    order = {"P0": 0, "P1": 1, "P2": 2}
    return a if order.get(a, 9) <= order.get(b, 9) else b


if __name__ == "__main__":
    raise SystemExit(main())

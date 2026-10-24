from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assured_rag.module_confusion import condition_failure_mass
from assured_rag.module_confusion_backend_compare import backend_label, failure_kind


DEFAULT_ERROR_ANALYSIS = Path("reports/operator_approved_error_analysis.json")
DEFAULT_AUDIT = Path("reports/mistral_paper_card_audit.json")
DEFAULT_BACKEND_JSONS = [
    Path("reports/operator_approved_mistral_rag.json"),
    Path("reports/operator_approved_3_5_rag.json"),
    Path("reports/operator_approved_4_1_nano_rag.json"),
    Path("reports/operator_approved_grok_rag.json"),
]

MODULE_TO_WEEK = {
    "M01-02": (1, "M01-02", "Introduction and AI safety framing"),
    "M03": (2, "M03", "Safety, security, and privacy"),
    "M04": (3, "M04", "Anomaly and fault detection"),
    "M05": (4, "M05", "Dataset shift and OOD behavior"),
    "M06": (5, "M06", "Verification and validation"),
    "M07": (6, "M07", "Testing and evaluation"),
    "M08": (7, "M08", "Interpretable AI"),
    "M09": (8, "M09", "Human-AI integration and oversight"),
    "M10": (9, "M10", "Policy/governance and responsible AI"),
    "M11": (10, "M11", "Adversarial autonomy"),
    "M12": (11, "M12", "Runtime assurance"),
    "M13": (12, "M13", "Safe reinforcement learning"),
    "M14": (13, "M14", "Legal, policy, and ethical implications"),
}

FAILURE_TO_BLIND_SPOTS = {
    "hard_stop_softening": [
        "verdict_risk_monotonicity_failure",
        "reject_decision_noncompliance",
        "operational_output_pathway_underweighting",
    ],
    "severe_decision_rule_incoherence": [
        "verdict_risk_monotonicity_failure",
        "reject_decision_noncompliance",
    ],
    "auditability_failure": [
        "remediation_as_evidence_confusion",
        "malformed_advisory_output",
        "operational_output_pathway_underweighting",
    ],
    "rag_overload_or_confusion": [
        "remediation_as_evidence_confusion",
        "safety_label_laundering_susceptibility",
    ],
    "unsupported_claim_metric_suspect": [
        "remediation_as_evidence_confusion",
        "malformed_advisory_output",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute module confusion to specific paper cards.")
    parser.add_argument("--error-analysis", default=str(DEFAULT_ERROR_ANALYSIS))
    parser.add_argument("--paper-card-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--backend-jsons", nargs="*", default=[str(path) for path in DEFAULT_BACKEND_JSONS])
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-week-plot", required=True)
    parser.add_argument("--output-heatmap", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    error_analysis = json.loads(Path(args.error_analysis).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.paper_card_audit).read_text(encoding="utf-8")) if Path(args.paper_card_audit).exists() else {}
    backend_runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.backend_jsons if Path(path).exists()]
    analysis = build_paper_attribution(error_analysis, audit, backend_runs)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_week_plot = Path(args.output_week_plot)
    output_heatmap = Path(args.output_heatmap)
    for path in [output_md, output_json, output_csv, output_week_plot, output_heatmap]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(analysis["evidence_records"], output_csv)
    plot_top_papers(analysis["paper_rows"][:15], output_week_plot)
    plot_week_paper_heatmap(analysis["within_week_rows"], output_heatmap)
    output_md.write_text(render_markdown(analysis, output_week_plot, output_heatmap), encoding="utf-8")

    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_week_plot}")
    print(f"Wrote {output_heatmap}")
    return 0


def build_paper_attribution(
    error_analysis: dict[str, Any],
    paper_audit: dict[str, Any],
    backend_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_by_blind_spot = {item["blind_spot_id"]: item for item in paper_audit.get("mappings", [])}
    packet_lookup = build_packet_lookup(backend_runs)
    records = []
    for condition in error_analysis.get("conditions", []):
        backend = backend_label(condition)
        fixture = condition.get("fixture", "unknown_fixture")
        parsed = float(condition.get("parsed_successes", 0) or 0)
        for failure in condition.get("classifications", []):
            if failure == "no_observed_failure":
                continue
            mass = condition_failure_mass(condition, failure, parsed)
            if mass <= 0:
                continue
            records.extend(packet_records(condition, backend, fixture, failure, mass, packet_lookup))
            records.extend(audit_mapping_records(condition, backend, fixture, failure, mass, audit_by_blind_spot))

    aggregated = aggregate_records(records)
    return {
        "title": "Paper-Level Confusion Attribution",
        "evidence_records": records,
        "paper_rows": aggregated["paper_rows"],
        "within_week_rows": aggregated["within_week_rows"],
        "backend_rows": aggregated["backend_rows"],
        "failure_type_rows": aggregated["failure_type_rows"],
        "sensitivity": build_sensitivity(records),
        "research_interpretation": build_research_interpretation(aggregated),
        "caveats": [
            "Week-level bins are coarse.",
            "Paper-level attribution depends on paper-card evidence mapping and saved evidence packets.",
            "Zero weight means not implicated by current saved experiments, not unimportant overall.",
        ],
    }


def build_packet_lookup(backend_runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    lookup = {}
    for run in backend_runs:
        backend = backend_label(run)
        fixture = run.get("fixture_name", "unknown_fixture")
        lookup[(backend, fixture)] = {
            item.get("evidence_id"): item
            for item in run.get("evidence_packet", {}).get("items", [])
            if item.get("evidence_id")
        }
    return lookup


def packet_records(
    condition: dict[str, Any],
    backend: str,
    fixture: str,
    failure: str,
    mass: float,
    packet_lookup: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    if condition.get("rag_mode") != "paper_cards":
        return []
    packet = packet_lookup.get((backend, fixture), {})
    if not packet:
        return []
    cited_ids = cited_evidence_ids(condition)
    if not cited_ids:
        if failure == "auditability_failure":
            cited_ids = list(packet)
        else:
            return []
    records = []
    weight = mass / max(len(cited_ids), 1)
    for evidence_id in cited_ids:
        item = packet.get(evidence_id)
        if not item:
            continue
        records.append(make_record(
            filename=item.get("filename", ""),
            chunk_id=item.get("chunk_id", ""),
            evidence_id=evidence_id,
            backend=backend,
            failure=failure,
            fixture=fixture,
            weight=weight,
            source="backend_evidence_packet",
            why="Evidence ID used by the external judge, or available but omitted in an auditability failure.",
        ))
    return records


def cited_evidence_ids(condition: dict[str, Any]) -> list[str]:
    seen = []
    for example in condition.get("representative_examples", []):
        for evidence_id in example.get("evidence_ids_used", []):
            if evidence_id and evidence_id.startswith("E") and evidence_id not in seen:
                seen.append(evidence_id)
    return seen


def audit_mapping_records(
    condition: dict[str, Any],
    backend: str,
    fixture: str,
    failure: str,
    mass: float,
    audit_by_blind_spot: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    blind_spots = FAILURE_TO_BLIND_SPOTS.get(failure, [])
    excerpts = []
    for blind_spot in blind_spots:
        excerpts.extend(audit_by_blind_spot.get(blind_spot, {}).get("paper_card_excerpts", []))
    if not excerpts:
        return []
    unique = dedupe_excerpts(excerpts)[:12]
    weight = mass / max(len(unique), 1)
    records = []
    for item in unique:
        records.append(make_record(
            filename=item.get("filename", item.get("paper_card_id", "")),
            chunk_id=item.get("chunk_id", ""),
            evidence_id=item.get("paper_card_id", ""),
            backend=backend,
            failure=failure,
            fixture=fixture,
            weight=weight,
            source="blind_spot_paper_card_mapping",
            why=f"Paper card mapped to blind spot support for `{failure}` via {', '.join(blind_spots)}.",
        ))
    return records


def dedupe_excerpts(excerpts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in excerpts:
        key = (item.get("filename", item.get("paper_card_id", "")), item.get("chunk_id", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def make_record(
    *,
    filename: str,
    chunk_id: str,
    evidence_id: str,
    backend: str,
    failure: str,
    fixture: str,
    weight: float,
    source: str,
    why: str,
) -> dict[str, Any]:
    identity = extract_paper_identity(filename)
    return {
        **identity,
        "paper_card_filename": filename,
        "chunk_id": chunk_id,
        "evidence_id": evidence_id,
        "backend": backend,
        "failure_type": failure,
        "failure_kind": failure_kind(failure),
        "fixture": fixture,
        "weight": round(weight, 6),
        "source": source,
        "why_implicated": why,
    }


def extract_paper_identity(filename: str) -> dict[str, Any]:
    base = Path(filename).name
    stem = re.sub(r"\.md$", "", base)
    stem = re.sub(r"\.readable$", "", stem)
    module_match = re.search(r"(M\d{2}(?:-\d{2})?)", stem)
    module_id = module_match.group(1) if module_match else "Unknown"
    week, normalized_module, module_name = MODULE_TO_WEEK.get(module_id, (14, module_id, "Project or uncategorized reading"))
    title = re.sub(r"^M\d{2}(?:-\d{2})?[_-]", "", stem)
    title = re.sub(r"[-_]pdf$", "", title)
    title = title.replace("_", " ").replace("-", " ").replace(".pdf", "")
    title = re.sub(r"\s+", " ", title).strip() or stem
    short = stem[:90]
    return {
        "week": week,
        "module_id": normalized_module,
        "module_name": module_name,
        "paper_title": title,
        "paper_short_name": short,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["weight"] for row in records)
    paper_weights: dict[str, float] = defaultdict(float)
    paper_meta: dict[str, dict[str, Any]] = {}
    week_totals: dict[int, float] = defaultdict(float)
    backend_weights: dict[tuple[str, str], float] = defaultdict(float)
    failure_weights: dict[tuple[str, str], float] = defaultdict(float)
    for row in records:
        key = row["paper_short_name"]
        paper_weights[key] += row["weight"]
        paper_meta[key] = row
        week_totals[row["week"]] += row["weight"]
        backend_weights[(row["backend"], key)] += row["weight"]
        failure_weights[(row["failure_type"], key)] += row["weight"]

    paper_rows = []
    for key, weight in sorted(paper_weights.items(), key=lambda item: item[1], reverse=True):
        meta = paper_meta[key]
        paper_rows.append({
            "paper_short_name": key,
            "paper_title": meta["paper_title"],
            "week": meta["week"],
            "module_id": meta["module_id"],
            "module_name": meta["module_name"],
            "weight": round(weight, 6),
            "probability": round(weight / total, 6) if total else 0.0,
            "main_backend": dominant_for(records, key, "backend"),
            "main_failure_type": dominant_for(records, key, "failure_type"),
            "why_implicated": meta["why_implicated"],
        })

    within_week_rows = []
    for row in paper_rows:
        week_total = week_totals[row["week"]]
        packed = dict(row)
        packed["probability_within_week"] = round(row["weight"] / week_total, 6) if week_total else 0.0
        within_week_rows.append(packed)

    return {
        "paper_rows": paper_rows,
        "within_week_rows": within_week_rows,
        "backend_rows": aggregate_pair_rows(backend_weights, paper_meta, "backend"),
        "failure_type_rows": aggregate_pair_rows(failure_weights, paper_meta, "failure_type"),
    }


def dominant_for(records: list[dict[str, Any]], paper_key: str, field: str) -> str:
    counter: Counter[str] = Counter()
    for row in records:
        if row["paper_short_name"] == paper_key:
            counter[row[field]] += row["weight"]
    return counter.most_common(1)[0][0] if counter else ""


def aggregate_pair_rows(weights: dict[tuple[str, str], float], meta: dict[str, dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    for (value, _paper), weight in weights.items():
        totals[value] += weight
    rows = []
    for (value, paper), weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        info = meta[paper]
        rows.append({
            dimension: value,
            "paper_short_name": paper,
            "paper_title": info["paper_title"],
            "week": info["week"],
            "module_id": info["module_id"],
            "weight": round(weight, 6),
            "probability": round(weight / totals[value], 6) if totals[value] else 0.0,
        })
    return rows


def build_sensitivity(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_backends": top_papers(records),
        "mistral_only": top_papers([row for row in records if row["backend"] == "Mistral"]),
        "exclude_mistral": top_papers([row for row in records if row["backend"] != "Mistral"]),
        "openai_plus_grok": top_papers([row for row in records if row["backend"].startswith("OpenAI") or row["backend"] == "Grok"]),
    }


def top_papers(records: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    aggregate = aggregate_records(records)["paper_rows"] if records else []
    return aggregate[:limit]


def build_research_interpretation(aggregated: dict[str, Any]) -> dict[str, Any]:
    important_weeks = {5, 6, 11, 13, 14}
    top_by_week = {
        str(week): [row for row in aggregated["within_week_rows"] if row["week"] == week][:5]
        for week in important_weeks
    }
    week11 = top_by_week.get("11", [])
    unique_week11 = {row["paper_short_name"] for row in week11}
    return {
        "top_by_important_week": top_by_week,
        "runtime_assurance_driven_by": "multiple papers" if len(unique_week11) > 1 else "one paper" if unique_week11 else "no current paper evidence",
        "recommended_first_reads": recommended_first_reads(aggregated["paper_rows"]),
        "summary": "Read the top Week 11 runtime-assurance cards for hard-stop enforcement, then the Week 6 safety-case/testing cards for evidence and auditability design.",
    }


def recommended_first_reads(paper_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in paper_rows:
        if row["week"] in {5, 6, 11, 13, 14} and row["paper_short_name"] not in seen:
            selected.append(row)
            seen.add(row["paper_short_name"])
        if len(selected) >= 8:
            break
    return selected


def plot_top_papers(paper_rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(paper_rows)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.38)))
    sns.barplot(data=df, y="paper_short_name", x="weight", hue="module_id", dodge=False, ax=ax)
    ax.set_title("Top Paper/Card Confusion Attribution", pad=14)
    ax.set_xlabel("Confusion weight")
    ax.set_ylabel("Paper/card")
    ax.legend(title="Module", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_week_paper_heatmap(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    top_names = [row["paper_short_name"] for row in sorted(rows, key=lambda item: item["weight"], reverse=True)[:20]]
    df = pd.DataFrame([row for row in rows if row["paper_short_name"] in top_names])
    pivot = df.pivot_table(index="paper_short_name", columns="week", values="weight", aggfunc="sum", fill_value=0.0)
    for week in range(1, 15):
        if week not in pivot.columns:
            pivot[week] = 0.0
    pivot = pivot[sorted(pivot.columns)]
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.4, ax=ax)
    ax.set_title("Paper/Card Confusion by Course Week", pad=14)
    ax.set_xlabel("Week")
    ax.set_ylabel("Paper/card")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "week",
        "module_id",
        "module_name",
        "paper_short_name",
        "paper_title",
        "paper_card_filename",
        "chunk_id",
        "evidence_id",
        "backend",
        "failure_type",
        "failure_kind",
        "fixture",
        "weight",
        "source",
        "why_implicated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def render_markdown(analysis: dict[str, Any], week_plot: Path, heatmap: Path) -> str:
    lines = [
        "# Paper-Level Confusion Attribution",
        "",
        "This report attributes saved model-confusion signals to specific paper cards. No API calls or browsing were performed.",
        "",
        "## Figures",
        "",
        f"- Top paper bar chart: `{week_plot.as_posix()}`",
        f"- Paper/week heatmap: `{heatmap.as_posix()}`",
        "",
        "## Top Papers Overall",
        "",
        "| Paper/card | Week | Weight | Probability | Main backend | Main failure type |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in analysis["paper_rows"][:15]:
        lines.append(
            f"| {row['paper_short_name']} | W{row['week']} / {row['module_id']} | {row['weight']:.3f} | "
            f"{row['probability']:.3f} | {row['main_backend']} | {row['main_failure_type']} |"
        )
    lines.extend(["", "## Within-Week Breakdown", ""])
    for week in [6, 11, 5, 14, 13]:
        rows = [row for row in analysis["within_week_rows"] if row["week"] == week][:8]
        lines.extend([f"### Week {week}", "", "| Paper/card | Weight | Probability within week | Main backend | Main failure type | Why implicated |", "|---|---:|---:|---|---|---|"])
        if not rows:
            lines.append("| No current paper evidence | 0 | 0 |  |  | Not implicated by current saved experiments |")
        for row in rows:
            lines.append(
                f"| {row['paper_short_name']} | {row['weight']:.3f} | {row['probability_within_week']:.3f} | "
                f"{row['main_backend']} | {row['main_failure_type']} | {row['why_implicated']} |"
            )
        lines.append("")
    lines.extend([
        "## Backend-Specific Paper Attribution",
        "",
        "| Backend | Top paper/card | Week | Weight | Failure type | Interpretation |",
        "|---|---|---:|---:|---|---|",
    ])
    for backend in sorted({row["backend"] for row in analysis["evidence_records"]}):
        rows = [row for row in analysis["backend_rows"] if row["backend"] == backend]
        if not rows:
            continue
        top = rows[0]
        failure = next((row["main_failure_type"] for row in analysis["paper_rows"] if row["paper_short_name"] == top["paper_short_name"]), "")
        lines.append(
            f"| {backend} | {top['paper_short_name']} | W{top['week']} / {top['module_id']} | "
            f"{top['weight']:.3f} | {failure} | {backend_interpretation(backend, failure)} |"
        )
    lines.extend([
        "",
        "## Which Papers Should Guide The Research Project?",
        "",
        analysis["research_interpretation"]["summary"],
        "",
        f"Runtime-assurance interest is driven by **{analysis['research_interpretation']['runtime_assurance_driven_by']}** in the current saved attribution.",
        "",
        "Recommended first reads:",
    ])
    for row in analysis["research_interpretation"]["recommended_first_reads"]:
        lines.append(f"- W{row['week']} / {row['module_id']}: `{row['paper_short_name']}`")
    lines.extend(["", "## Sensitivity", ""])
    for name, rows in analysis["sensitivity"].items():
        lines.append(f"### {name}")
        for row in rows[:5]:
            lines.append(f"- W{row['week']} `{row['paper_short_name']}`: {row['probability']:.3f}")
        lines.append("")
    lines.extend(["## Caveats", ""])
    for caveat in analysis["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines).strip() + "\n"


def backend_interpretation(backend: str, failure: str) -> str:
    if backend == "Mistral":
        return "Runtime-assurance hard-stop propagation is the key stress case."
    if "gpt-3.5" in backend:
        return "RAG packet size and hard-stop ordering should be tested."
    if "gpt-4.1-nano" in backend:
        return "Metric calibration is more important than label repair."
    if backend == "Grok":
        return "Evidence-ID compliance is the key auditability issue."
    return f"Review `{failure}` against the cited paper card."


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assured_rag.module_confusion import (
    COURSE_WEEKS,
    FAILURE_SEVERITY,
    FAILURE_WEEK_WEIGHTS,
    condition_failure_mass,
    normalize_weeks,
)


DEFAULT_ERROR_ANALYSIS = Path("reports/operator_approved_error_analysis.json")
FAILURE_TYPES = [
    "hard_stop_softening",
    "auditability_failure",
    "rag_overload_or_confusion",
    "unsupported_claim_metric_suspect",
    "severe_decision_rule_incoherence",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-backend module confusion distributions.")
    parser.add_argument("--input-json", default=str(DEFAULT_ERROR_ANALYSIS))
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-heatmap", required=True)
    parser.add_argument("--github-strategy", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    analysis = build_backend_breakdown(source)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    output_heatmap = Path(args.output_heatmap)
    github_strategy = Path(args.github_strategy)
    for path in [output_md, output_json, output_csv, output_png, output_heatmap, github_strategy]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(analysis["backend_week_rows"], output_csv)
    plot_backend_lines(analysis["backend_week_rows"], output_png)
    plot_failure_heatmap(analysis["failure_type_matrix_rows"], output_heatmap)
    output_md.write_text(render_markdown(analysis, output_png, output_heatmap), encoding="utf-8")
    github_strategy.write_text(render_github_strategy(analysis), encoding="utf-8")

    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_heatmap}")
    print(f"Wrote {github_strategy}")
    return 0


def build_backend_breakdown(error_analysis: dict[str, Any]) -> dict[str, Any]:
    backend_week_weights: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    backend_failure_masses: dict[str, Counter[str]] = defaultdict(Counter)
    backend_failure_weeks: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    backend_fixtures: dict[str, Counter[str]] = defaultdict(Counter)
    backend_failure_kinds: dict[str, set[str]] = defaultdict(set)

    for condition in error_analysis.get("conditions", []):
        backend = backend_label(condition)
        parsed = float(condition.get("parsed_successes", 0) or 0)
        fixture = condition.get("fixture", "unknown_fixture")
        for failure in condition.get("classifications", []):
            if failure == "no_observed_failure":
                continue
            mass = condition_failure_mass(condition, failure, parsed)
            backend_failure_masses[backend][failure] += mass
            backend_fixtures[backend][fixture] += 1
            backend_failure_kinds[backend].add(failure_kind(failure))
            for week, share in FAILURE_WEEK_WEIGHTS.get(failure, {}).items():
                contribution = mass * share
                backend_week_weights[backend][week] += contribution
                backend_failure_weeks[backend][failure][week] += contribution

    backend_distributions = []
    backend_week_rows = []
    for backend in sorted(backend_week_weights):
        weeks = normalize_weeks(backend_week_weights[backend])
        dominant_failure_by_week = dominant_failures_for_backend(backend, backend_failure_weeks[backend])
        for row in weeks:
            row["backend"] = backend
            row["dominant_failure_types"] = dominant_failure_by_week.get(row["week"], [])
            row["representative_fixture"] = most_common(backend_fixtures[backend])
            row["failure_kind"] = ", ".join(sorted(backend_failure_kinds[backend])) or "unmeasured"
            backend_week_rows.append(row)
        backend_distributions.append({
            "backend": backend,
            "weeks": weeks,
            "top_week": max(weeks, key=lambda item: item["probability"]),
            "failure_masses": dict(backend_failure_masses[backend]),
            "top_failure_type": most_common(backend_failure_masses[backend]),
            "representative_fixture": most_common(backend_fixtures[backend]),
            "failure_kind": ", ".join(sorted(backend_failure_kinds[backend])) or "unmeasured",
        })

    openai_rows = build_openai_aggregate(backend_distributions)
    backend_week_rows.extend(openai_rows["backend_week_rows"])
    if openai_rows["distribution"]:
        backend_distributions.append(openai_rows["distribution"])

    gpt55_status = {
        "measured": any("gpt-5.5" in item["backend"].lower() for item in backend_distributions),
        "message": "GPT-5.5 is unmeasured in this experiment. Do not infer that it has no issues.",
    }
    failure_type_matrix_rows = build_failure_type_matrix(backend_distributions)
    runtime_vs_verification = runtime_verification_summary(backend_distributions, failure_type_matrix_rows)
    return {
        "title": "Per-LLM Assurance Confusion by Course Week",
        "backend_distributions": backend_distributions,
        "backend_week_rows": sorted(backend_week_rows, key=lambda row: (row["backend"], row["week"])),
        "failure_type_matrix_rows": failure_type_matrix_rows,
        "runtime_vs_verification": runtime_vs_verification,
        "gpt_5_5_status": gpt55_status,
        "research_routing_table": research_routing_table(backend_distributions),
        "notes": [
            "All results are computed from saved artifacts; no API calls are made.",
            "Backend outputs are advisory only; deterministic assurance verdicts remain authoritative.",
        ],
    }


def backend_label(condition: dict[str, Any]) -> str:
    provider = condition.get("provider", "unknown")
    model = condition.get("model", "unknown")
    if provider == "openai":
        return f"OpenAI {model}"
    if provider == "mistral":
        return "Mistral"
    if provider == "grok":
        return "Grok"
    return f"{provider}/{model}"


def failure_kind(failure: str) -> str:
    return {
        "hard_stop_softening": "label failure",
        "severe_decision_rule_incoherence": "label failure",
        "rag_overload_or_confusion": "RAG overload",
        "auditability_failure": "auditability failure",
        "unsupported_claim_metric_suspect": "metric artifact",
    }.get(failure, "other")


def most_common(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def dominant_failures_for_backend(backend: str, failure_weeks: dict[str, dict[int, float]]) -> dict[int, list[str]]:
    del backend
    by_week: dict[int, Counter[str]] = defaultdict(Counter)
    for failure, week_weights in failure_weeks.items():
        for week, weight in week_weights.items():
            if weight > 0:
                by_week[week][failure] += weight
    return {
        week: [failure for failure, _ in counts.most_common(2)]
        for week, counts in by_week.items()
    }


def build_openai_aggregate(distributions: list[dict[str, Any]]) -> dict[str, Any]:
    openai = [item for item in distributions if item["backend"].startswith("OpenAI ")]
    if not openai:
        return {"distribution": None, "backend_week_rows": []}
    weights: dict[int, float] = defaultdict(float)
    failures: Counter[str] = Counter()
    fixtures: Counter[str] = Counter()
    kinds: set[str] = set()
    for item in openai:
        for row in item["weeks"]:
            weights[row["week"]] += row["weight"]
        failures.update(item["failure_masses"])
        if item["representative_fixture"]:
            fixtures[item["representative_fixture"]] += 1
        if item["failure_kind"]:
            kinds.update(part.strip() for part in item["failure_kind"].split(","))
    weeks = normalize_weeks(weights)
    rows = []
    for row in weeks:
        row["backend"] = "OpenAI aggregate"
        row["dominant_failure_types"] = [failures.most_common(1)[0][0]] if failures else []
        row["representative_fixture"] = most_common(fixtures)
        row["failure_kind"] = ", ".join(sorted(kinds))
        rows.append(row)
    return {
        "backend_week_rows": rows,
        "distribution": {
            "backend": "OpenAI aggregate",
            "weeks": weeks,
            "top_week": max(weeks, key=lambda item: item["probability"]),
            "failure_masses": dict(failures),
            "top_failure_type": most_common(failures),
            "representative_fixture": most_common(fixtures),
            "failure_kind": ", ".join(sorted(kinds)),
        },
    }


def build_failure_type_matrix(distributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in distributions:
        masses = item.get("failure_masses", {})
        for failure in FAILURE_TYPES:
            rows.append({
                "backend": item["backend"],
                "failure_type": failure,
                "weighted_mass": round(float(masses.get(failure, 0.0)), 6),
            })
    return rows


def runtime_verification_summary(distributions: list[dict[str, Any]], matrix_rows: list[dict[str, Any]]) -> dict[str, Any]:
    focus_weeks = {5: "Verification and validation", 6: "Testing and evaluation", 11: "Runtime assurance", 14: "Project assurance synthesis"}
    week_by_backend = {}
    for item in distributions:
        week_by_backend[item["backend"]] = {
            row["week"]: row["probability"]
            for row in item["weeks"]
            if row["week"] in focus_weeks
        }
    runtime_backend = max(distributions, key=lambda item: week_probability(item, 11), default=None)
    testing_backend = max(distributions, key=lambda item: week_probability(item, 6), default=None)
    failure_week_scores = {}
    for failure, shares in FAILURE_WEEK_WEIGHTS.items():
        failure_week_scores[failure] = {
            "runtime_assurance_share": shares.get(11, 0.0),
            "testing_evaluation_share": shares.get(6, 0.0),
        }
    return {
        "focus_weeks": focus_weeks,
        "week_by_backend": week_by_backend,
        "most_runtime_assurance_confused_backend": runtime_backend["backend"] if runtime_backend else "",
        "most_testing_evaluation_confused_backend": testing_backend["backend"] if testing_backend else "",
        "failure_type_most_runtime_assurance": max(failure_week_scores, key=lambda key: failure_week_scores[key]["runtime_assurance_share"]),
        "failure_type_most_testing_evaluation": max(failure_week_scores, key=lambda key: failure_week_scores[key]["testing_evaluation_share"]),
        "interpretation": "Runtime assurance is the sharper research target for hard-stop enforcement; testing/evaluation remains the broader target for evidence packet design, citation compliance, and metric calibration.",
    }


def week_probability(distribution: dict[str, Any], week: int) -> float:
    for row in distribution.get("weeks", []):
        if row["week"] == week:
            return float(row["probability"])
    return 0.0


def research_routing_table(distributions: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for item in distributions:
        if item["backend"] == "OpenAI aggregate":
            continue
        top_week = item["top_week"]
        rows.append({
            "backend": item["backend"],
            "main_blindspot_week": f"Week {top_week['week']} / {top_week['module_id']}",
            "main_failure_type": item["top_failure_type"],
            "research_implication": research_implication(item["top_failure_type"], top_week["week"], item["backend"]),
        })
    return rows


def research_implication(failure: str, week: int, backend: str) -> str:
    if failure == "hard_stop_softening":
        return "Study runtime-assurance hard stops and consistency validators."
    if failure == "rag_overload_or_confusion":
        return "Minimize evidence packets and put hard-stop rules before citations."
    if failure == "auditability_failure":
        return "Enforce evidence-ID compliance separately from label correctness."
    if failure == "unsupported_claim_metric_suspect":
        return "Calibrate audit metrics so caveats are not counted as unsupported facts."
    return f"Route {backend} failures through Week {week} assurance-case review."


def plot_backend_lines(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df, x="week", y="probability", hue="backend", marker="o", ax=ax)
    ax.set_title("Per-LLM Assurance Confusion by Course Week", pad=18)
    ax.set_xlabel("Week number")
    ax.set_ylabel("Normalized confusion probability")
    ax.set_xticks(list(range(1, 15)))
    ax.legend(title="Backend/model", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_failure_heatmap(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="backend", columns="failure_type", values="weighted_mass").fillna(0.0)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Per-LLM Failure-Type Mass", pad=16)
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Backend/model")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "backend",
        "week",
        "module_id",
        "module_name",
        "weight",
        "probability",
        "research_priority",
        "dominant_failure_types",
        "representative_fixture",
        "failure_kind",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["dominant_failure_types"] = ";".join(clean.get("dominant_failure_types", []))
            writer.writerow({field: clean.get(field, "") for field in fields})


def render_markdown(analysis: dict[str, Any], plot_path: Path, heatmap_path: Path) -> str:
    lines = [
        "# Per-LLM Blindspot Breakdown",
        "",
        "All rows are computed from saved backend experiments. No new model calls or browsing were performed.",
        "",
        "## Figures",
        "",
        f"- Per-backend week plot: `{plot_path.as_posix()}`",
        f"- Failure-type heatmap: `{heatmap_path.as_posix()}`",
        "",
        "## Backend Summary",
        "",
        "| Backend | Top week | Probability | Top failure type | Failure kind | Representative fixture |",
        "|---|---|---:|---|---|---|",
    ]
    for item in analysis["backend_distributions"]:
        top = item["top_week"]
        lines.append(
            f"| {item['backend']} | Week {top['week']} / {top['module_id']} {top['module_name']} | "
            f"{top['probability']:.3f} | {item['top_failure_type']} | {item['failure_kind']} | {item['representative_fixture']} |"
        )
    lines.extend([
        "",
        "## Runtime Assurance vs Verification",
        "",
        f"- Most Week 11 / runtime-assurance confusion: **{analysis['runtime_vs_verification']['most_runtime_assurance_confused_backend']}**.",
        f"- Most Week 6 / testing-evaluation confusion: **{analysis['runtime_vs_verification']['most_testing_evaluation_confused_backend']}**.",
        f"- Failure type most tied to runtime assurance: `{analysis['runtime_vs_verification']['failure_type_most_runtime_assurance']}`.",
        f"- Failure type most tied to testing/evaluation: `{analysis['runtime_vs_verification']['failure_type_most_testing_evaluation']}`.",
        f"- Interpretation: {analysis['runtime_vs_verification']['interpretation']}",
        "",
        "## GPT-5.5 Status",
        "",
        analysis["gpt_5_5_status"]["message"] if not analysis["gpt_5_5_status"]["measured"] else "GPT-5.5 has measured rows in this experiment.",
        "",
        "## Research Routing Table",
        "",
        "| Backend | Main blindspot week | Main failure type | Research implication |",
        "|---|---:|---|---|",
    ])
    for row in analysis["research_routing_table"]:
        lines.append(
            f"| {row['backend']} | {row['main_blindspot_week']} | {row['main_failure_type']} | {row['research_implication']} |"
        )
    lines.extend(["", "## Per-Backend Week Distributions", ""])
    for item in analysis["backend_distributions"]:
        lines.extend([f"### {item['backend']}", "", "| Week | Module | Probability | Dominant failure types |", "|---:|---|---:|---|"])
        for row in item["weeks"]:
            if row["probability"] > 0:
                lines.append(
                    f"| {row['week']} | {row['module_id']} {row['module_name']} | {row['probability']:.3f} | "
                    f"{', '.join(next((r['dominant_failure_types'] for r in analysis['backend_week_rows'] if r['backend'] == item['backend'] and r['week'] == row['week']), []))} |"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_github_strategy(analysis: dict[str, Any]) -> str:
    categories = [
        {
            "name": "runtime assurance / Simplex / safety monitor / fallback controller repositories",
            "week": "Week 11 / M12",
            "addresses": "Mistral hard-stop softening and runtime-assurance enforcement.",
            "queries": [
                "runtime assurance Simplex architecture autonomous systems GitHub",
                "safety monitor fallback controller robotics GitHub",
                "runtime assurance autonomous vehicle controller repository",
            ],
        },
        {
            "name": "testing and evaluation harnesses for autonomous systems",
            "week": "Week 6 / M07",
            "addresses": "GPT-3.5 RAG overload and general evaluation metric design.",
            "queries": [
                "autonomous systems testing evaluation harness GitHub",
                "simulation based testing autonomous vehicles safety GitHub",
                "adversarial test generation autonomous systems repository",
            ],
        },
        {
            "name": "safety-case / assurance-case / GSN tooling",
            "week": "Week 14 / Project",
            "addresses": "Compiler-level claim/evidence trace and hard-stop auditability.",
            "queries": [
                "assurance case GSN tooling GitHub",
                "safety case argument evidence repository",
                "goal structuring notation safety case Python",
            ],
        },
        {
            "name": "LLM evaluation frameworks with structured judge outputs",
            "week": "Week 6 / M07",
            "addresses": "External judge consistency, unsupported-claim calibration, and schema validation.",
            "queries": [
                "LLM judge structured output evaluation GitHub",
                "LLM eval schema validation citations repository",
                "LLM as judge consistency checker GitHub",
            ],
        },
        {
            "name": "RAG citation-validation / evidence-grounding tools",
            "week": "Week 6 / M07 and Week 13 / M14",
            "addresses": "Grok evidence-ID omission and GPT-4.1-nano unsupported-claim metric artifacts.",
            "queries": [
                "RAG citation validation evidence grounding GitHub",
                "retrieval augmented generation citation checker repository",
                "RAG attribution evaluation framework GitHub",
            ],
        },
        {
            "name": "human-in-the-loop autonomy evaluation projects",
            "week": "Week 11 / M12 and Week 14 / Project",
            "addresses": "Operator approval as oversight rather than containment.",
            "queries": [
                "human in the loop autonomous systems evaluation GitHub",
                "operator approval autonomous system safety monitor repository",
                "human oversight runtime assurance autonomy GitHub",
            ],
        },
    ]
    lines = [
        "# GitHub Search Strategy From Blindspots",
        "",
        "Do not treat this as a search result. It is the offline search plan produced from the per-LLM blindspot map.",
        "",
        "## Why These Categories",
        "",
        "The saved experiments concentrate confusion around testing/evaluation, runtime assurance, verification, and project-level assurance-case synthesis. The next GitHub pass should therefore look for codebases that expose these concepts at implementation level.",
        "",
    ]
    for category in categories:
        lines.extend([
            f"## {category['name']}",
            "",
            f"- Course support: {category['week']}",
            f"- Backend failure addressed: {category['addresses']}",
            "- Example queries:",
        ])
        for query in category["queries"]:
            lines.append(f"  - `{query}`")
        lines.append("")
    lines.extend(["## Backend Routing", ""])
    for row in analysis["research_routing_table"]:
        lines.append(
            f"- {row['backend']}: start with {row['main_blindspot_week']} because `{row['main_failure_type']}` suggests {row['research_implication']}"
        )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

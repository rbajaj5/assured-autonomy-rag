from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from assured_rag.module_confusion import COURSE_WEEKS, FAILURE_WEEK_WEIGHTS, condition_failure_mass, normalize_weeks
from assured_rag.module_confusion_backend_compare import backend_label


DEFAULT_ERROR_ANALYSIS = Path("reports/operator_approved_error_analysis.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for module-confusion distributions.")
    parser.add_argument("--input-json", default=str(DEFAULT_ERROR_ANALYSIS))
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-heatmap", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    analysis = build_sensitivity_analysis(source)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    output_heatmap = Path(args.output_heatmap)
    for path in [output_md, output_json, output_csv, output_png, output_heatmap]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(analysis["distribution_week_rows"], output_csv)
    plot_sensitivity_lines(analysis["distribution_week_rows"], output_png)
    plot_sensitivity_heatmap(analysis["distribution_week_rows"], output_heatmap)
    output_md.write_text(render_markdown(analysis, output_png, output_heatmap), encoding="utf-8")

    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_heatmap}")
    return 0


def build_sensitivity_analysis(error_analysis: dict[str, Any]) -> dict[str, Any]:
    conditions = error_analysis.get("conditions", [])
    specs = sensitivity_specs()
    distributions = []
    distribution_week_rows = []
    for name, predicate in specs.items():
        selected = [condition for condition in conditions if predicate(condition)]
        weeks = compute_distribution(selected)
        distribution = {
            "name": name,
            "condition_count": len(selected),
            "backends": sorted({backend_label(condition) for condition in selected}),
            "weeks": weeks,
            "top_weeks": sorted(weeks, key=lambda row: row["probability"], reverse=True)[:3],
            "week_11_probability": probability_for_week(weeks, 11),
        }
        distributions.append(distribution)
        for row in weeks:
            packed = dict(row)
            packed["distribution"] = name
            distribution_week_rows.append(packed)

    leave_one_out = build_leave_one_out(conditions)
    interpretation = interpret(distributions)
    return {
        "title": "Module Confusion Sensitivity Analysis",
        "distributions": distributions,
        "distribution_week_rows": distribution_week_rows,
        "runtime_assurance_sensitivity": runtime_sensitivity(distributions),
        "leave_one_backend_out": leave_one_out,
        "interpretation": interpretation,
        "research_implication": research_implication(interpretation),
        "notes": [
            "Mistral is not removed silently; it is compared explicitly against other backend subsets.",
            "All calculations use saved artifacts only. No API calls are made.",
        ],
    }


def sensitivity_specs() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "all_backends": lambda condition: True,
        "mistral_only": lambda condition: condition.get("provider") == "mistral",
        "exclude_mistral": lambda condition: condition.get("provider") != "mistral",
        "openai_only": lambda condition: condition.get("provider") == "openai",
        "openai_plus_grok": lambda condition: condition.get("provider") in {"openai", "grok"},
        "weak_openai_only": lambda condition: condition.get("provider") == "openai"
        and condition.get("model") in {"gpt-3.5-turbo", "gpt-4.1-nano"},
    }


def compute_distribution(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights: dict[int, float] = defaultdict(float)
    for condition in conditions:
        parsed = float(condition.get("parsed_successes", 0) or 0)
        for failure in condition.get("classifications", []):
            if failure == "no_observed_failure":
                continue
            mass = condition_failure_mass(condition, failure, parsed)
            for week, share in FAILURE_WEEK_WEIGHTS.get(failure, {}).items():
                weights[week] += mass * share
    return normalize_weeks(weights)


def build_leave_one_out(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_weeks = compute_distribution(conditions)
    all_week_11 = probability_for_week(all_weeks, 11)
    backends = sorted({backend_label(condition) for condition in conditions})
    rows = []
    for backend in backends:
        selected = [condition for condition in conditions if backend_label(condition) != backend]
        weeks = compute_distribution(selected)
        week_11 = probability_for_week(weeks, 11)
        top = sorted(weeks, key=lambda row: row["probability"], reverse=True)[:3]
        rows.append({
            "removed_backend": backend,
            "remaining_condition_count": len(selected),
            "week_11_probability": week_11,
            "week_11_delta_vs_all": round(week_11 - all_week_11, 6),
            "top_weeks": top,
        })
    return rows


def runtime_sensitivity(distributions: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {"all_backends", "mistral_only", "exclude_mistral", "openai_plus_grok"}
    return {
        item["name"]: {
            "week_11_probability": item["week_11_probability"],
            "top_weeks": item["top_weeks"],
            "backends": item["backends"],
        }
        for item in distributions
        if item["name"] in wanted
    }


def interpret(distributions: list[dict[str, Any]]) -> str:
    by_name = {item["name"]: item for item in distributions}
    all_week_11 = by_name.get("all_backends", {}).get("week_11_probability", 0.0)
    no_mistral_week_11 = by_name.get("exclude_mistral", {}).get("week_11_probability", 0.0)
    mistral_week_11 = by_name.get("mistral_only", {}).get("week_11_probability", 0.0)
    no_mistral_top = by_name.get("exclude_mistral", {}).get("top_weeks", [{}])[0].get("week")
    if mistral_week_11 >= 0.30 and no_mistral_week_11 < 0.15:
        return "runtime-assurance hard-stop confusion is Mistral-concentrated in current experiments."
    if no_mistral_week_11 >= 0.20:
        return "runtime-assurance hard-stop confusion appears shared across backends."
    if no_mistral_top == 6:
        return "after removing Mistral, confusion is primarily testing/evaluation and RAG auditability."
    return f"Week 11 drops from {all_week_11:.3f} to {no_mistral_week_11:.3f} without Mistral; interpret as mixed sensitivity."


def research_implication(interpretation: str) -> str:
    if "Mistral-concentrated" in interpretation:
        return "Project framing should be backend-aware hard-stop enforcement: Mistral is the stress case for runtime-assurance verdict propagation, while other backends stress auditability and metric calibration."
    if "shared" in interpretation:
        return "Project framing can target general LLM runtime-assurance reasoning across models."
    if "testing/evaluation" in interpretation:
        return "Project framing should emphasize testing/evaluation and RAG auditability after isolating Mistral-specific runtime-assurance failures."
    return "Project framing should keep both runtime assurance and testing/evaluation in scope until more fixtures are measured."


def probability_for_week(weeks: list[dict[str, Any]], week: int) -> float:
    for row in weeks:
        if row["week"] == week:
            return float(row["probability"])
    return 0.0


def plot_sensitivity_lines(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df, x="week", y="probability", hue="distribution", marker="o", ax=ax)
    ax.set_title("Module Confusion Sensitivity by Backend Subset", pad=18)
    ax.set_xlabel("Week number")
    ax.set_ylabel("Confusion probability")
    ax.set_xticks(list(range(1, 15)))
    ax.legend(title="Distribution", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_sensitivity_heatmap(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="distribution", columns="week", values="probability").fillna(0.0)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Module Confusion Sensitivity Heatmap", pad=16)
    ax.set_xlabel("Week number")
    ax.set_ylabel("Backend subset")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["distribution", "week", "module_id", "module_name", "weight", "probability", "research_priority"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_markdown(analysis: dict[str, Any], png_path: Path, heatmap_path: Path) -> str:
    lines = [
        "# Module Confusion Sensitivity Analysis",
        "",
        "This report compares module-confusion distributions with and without Mistral. No API calls were made.",
        "",
        "## Figures",
        "",
        f"- Sensitivity line plot: `{png_path.as_posix()}`",
        f"- Sensitivity heatmap: `{heatmap_path.as_posix()}`",
        "",
        "## Runtime Assurance Sensitivity",
        "",
        "| Distribution | Week 11 probability | Top weeks | Backends |",
        "|---|---:|---|---|",
    ]
    for name, item in analysis["runtime_assurance_sensitivity"].items():
        top = ", ".join(f"W{row['week']} {row['probability']:.3f}" for row in item["top_weeks"])
        lines.append(f"| {name} | {item['week_11_probability']:.3f} | {top} | {', '.join(item['backends'])} |")
    lines.extend([
        "",
        "## Leave-One-Backend-Out",
        "",
        "| Removed backend | Week 11 probability | Delta vs all | Top weeks after removal |",
        "|---|---:|---:|---|",
    ])
    for row in analysis["leave_one_backend_out"]:
        top = ", ".join(f"W{item['week']} {item['probability']:.3f}" for item in row["top_weeks"])
        lines.append(
            f"| {row['removed_backend']} | {row['week_11_probability']:.3f} | "
            f"{row['week_11_delta_vs_all']:.3f} | {top} |"
        )
    lines.extend([
        "",
        "## Distribution Details",
        "",
        "| Distribution | Backends | Condition count | Top weeks | Week 11 |",
        "|---|---|---:|---|---:|",
    ])
    for item in analysis["distributions"]:
        top = ", ".join(f"W{row['week']} {row['probability']:.3f}" for row in item["top_weeks"])
        lines.append(
            f"| {item['name']} | {', '.join(item['backends'])} | {item['condition_count']} | "
            f"{top} | {item['week_11_probability']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        analysis["interpretation"],
        "",
        "## Research Implication",
        "",
        analysis["research_implication"],
    ])
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

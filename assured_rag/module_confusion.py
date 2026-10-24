from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ANALYSIS = Path("reports/operator_approved_error_analysis.json")

COURSE_WEEKS = [
    (1, "M01-02", "Introduction and AI safety framing"),
    (2, "M03", "Safety, security, and privacy"),
    (3, "M04", "Anomaly and fault detection"),
    (4, "M05", "Dataset shift and OOD behavior"),
    (5, "M06", "Verification and validation"),
    (6, "M07", "Testing and evaluation"),
    (7, "M08", "Interpretable AI"),
    (8, "M09", "Adversarial inputs and robustness"),
    (9, "M10", "Multi-agent systems and cooperative control"),
    (10, "M11", "Cyberphysical security and privacy"),
    (11, "M12", "Runtime assurance"),
    (12, "M13", "Safe reinforcement learning"),
    (13, "M14", "Legal, policy, and ethical implications"),
    (14, "Project", "Assurance case synthesis and project integration"),
]

FAILURE_WEEK_WEIGHTS = {
    "severe_decision_rule_incoherence": {11: 0.35, 6: 0.25, 5: 0.20, 14: 0.20},
    "hard_stop_softening": {11: 0.40, 6: 0.25, 5: 0.15, 12: 0.10, 14: 0.10},
    "auditability_failure": {6: 0.35, 5: 0.20, 13: 0.20, 14: 0.25},
    "rag_overload_or_confusion": {4: 0.25, 6: 0.35, 1: 0.20, 14: 0.20},
    "unsupported_claim_metric_suspect": {1: 0.20, 5: 0.15, 6: 0.35, 13: 0.20, 14: 0.10},
}

FAILURE_SEVERITY = {
    "severe_decision_rule_incoherence": 4.0,
    "hard_stop_softening": 3.0,
    "rag_overload_or_confusion": 2.0,
    "auditability_failure": 1.5,
    "unsupported_claim_metric_suspect": 1.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 14-week module confusion histogram.")
    parser.add_argument("--input-json", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    analysis = build_module_confusion(source)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    for path in [output_json, output_csv, output_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(analysis["weeks"], output_csv)

    png_path = None
    if args.output_png:
        png_path = Path(args.output_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plot_histogram(analysis["weeks"], png_path)

    output_md.write_text(render_markdown(analysis, png_path), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    if png_path:
        print(f"Wrote {png_path}")
    return 0


def build_module_confusion(error_analysis: dict[str, Any]) -> dict[str, Any]:
    week_weights: dict[int, float] = defaultdict(float)
    backend_weights: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    failure_weights: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for condition in error_analysis.get("conditions", []):
        backend = f"{condition.get('provider')}/{condition.get('model')}"
        parsed = float(condition.get("parsed_successes", 0) or 0)
        for failure in condition.get("classifications", []):
            if failure == "no_observed_failure":
                continue
            mass = condition_failure_mass(condition, failure, parsed)
            for week, share in FAILURE_WEEK_WEIGHTS.get(failure, {}).items():
                contribution = mass * share
                week_weights[week] += contribution
                backend_weights[backend][week] += contribution
                failure_weights[failure][week] += contribution

    weeks = normalize_weeks(week_weights)
    return {
        "title": "LLM Assurance Confusion by Course Week",
        "source": "operator_approved_error_analysis",
        "weeks": weeks,
        "backend_histograms": {
            backend: normalize_weeks(weights)
            for backend, weights in sorted(backend_weights.items())
        },
        "failure_type_histograms": {
            failure: normalize_weeks(weights)
            for failure, weights in sorted(failure_weights.items())
        },
        "notes": [
            "Weights are severity-weighted confusion mass inferred from saved backend error classifications.",
            "The histogram is diagnostic, not a claim that the course week itself is flawed.",
            "All calculations use offline report artifacts; no API calls are made.",
        ],
    }


def condition_failure_mass(condition: dict[str, Any], failure: str, parsed: float) -> float:
    severity = FAILURE_SEVERITY.get(failure, 1.0)
    if failure == "hard_stop_softening":
        rate = float(condition.get("revise_high_frequency", 0.0)) + float(condition.get("revise_unacceptable_frequency", 0.0))
    elif failure == "rag_overload_or_confusion":
        rate = max(float(condition.get("deterministic_disagreement_rate", 0.0)), 0.01)
    elif failure == "auditability_failure":
        rate = 1.0 - float(condition.get("evidence_citation_rate", 0.0))
    elif failure == "unsupported_claim_metric_suspect":
        rate = float(condition.get("unsupported_claim_rate", 0.0))
    elif failure == "severe_decision_rule_incoherence":
        rate = float(condition.get("approve_unacceptable_frequency", 0.0))
    else:
        rate = float(condition.get("deterministic_disagreement_rate", 0.0))
    return max(parsed * rate * severity, 0.0)


def normalize_weeks(weights: dict[int, float]) -> list[dict[str, Any]]:
    total = sum(weights.values())
    rows = []
    for week, module_id, module_name in COURSE_WEEKS:
        weight = round(float(weights.get(week, 0.0)), 6)
        probability = round(weight / total, 6) if total else 0.0
        rows.append({
            "week": week,
            "module_id": module_id,
            "module_name": module_name,
            "weight": weight,
            "probability": probability,
            "research_priority": research_priority(probability),
        })
    return rows


def research_priority(probability: float) -> str:
    if probability >= 0.20:
        return "P0"
    if probability >= 0.10:
        return "P1"
    if probability > 0:
        return "P2"
    return "P3"


def plot_histogram(weeks: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(weeks, columns=["week", "module_id", "module_name", "weight", "probability", "research_priority"])
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="week", y="probability", color="#3B82F6", ax=ax)
    ax.set_title("LLM Assurance Confusion by Course Week", pad=18)
    ax.set_xlabel("Week number")
    ax.set_ylabel("Confusion probability")
    ax.set_ylim(0, max([row["probability"] for row in weeks] + [0.05]) * 1.2)
    ax.text(
        0.5,
        -0.22,
        "Higher bars indicate course weeks whose concepts are implicated in model failures.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    for index, row in enumerate(weeks):
        if row["probability"] > 0:
            ax.text(index, row["probability"], f"{row['probability']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["week", "module_id", "module_name", "weight", "probability", "research_priority"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(analysis: dict[str, Any], png_path: Path | None = None) -> str:
    lines = [
        "# LLM Assurance Confusion by Course Week",
        "",
        "This report maps saved external-judge failure modes onto the 14-week assured-autonomy course structure.",
        "The ASCII histogram remains the quick terminal-readable view; the PNG gives the same distribution as a conventional chart.",
        "",
    ]
    if png_path:
        lines.extend([
            "## Seaborn Figure",
            "",
            f"Generated figure: `{png_path.as_posix()}`",
            "",
        ])
    lines.extend([
        "## 14-Week ASCII Histogram",
        "",
        "```text",
        *ascii_histogram(analysis["weeks"]),
        "```",
        "",
        "## 14-Week Table",
        "",
        "| Week | Module | Module name | Weight | Probability | Priority |",
        "|---:|---|---|---:|---:|---|",
    ])
    for row in analysis["weeks"]:
        lines.append(
            f"| {row['week']} | {row['module_id']} | {row['module_name']} | "
            f"{row['weight']:.3f} | {row['probability']:.3f} | {row['research_priority']} |"
        )
    lines.extend(["", "## Backend-Specific Histograms", ""])
    for backend, rows in analysis["backend_histograms"].items():
        lines.extend([f"### {backend}", "", "```text", *ascii_histogram(rows), "```", ""])
    lines.extend(["## Failure-Type-Specific Histograms", ""])
    for failure, rows in analysis["failure_type_histograms"].items():
        lines.extend([f"### {failure}", "", "```text", *ascii_histogram(rows), "```", ""])
    lines.extend([
        "## Research Interpretation",
        "",
        "The dominant weeks are the ones that teach decision rules, testing/evaluation, runtime assurance, and project-level assurance-case synthesis.",
        "That is exactly where the operator-approved-actions fixture is trying to stress the models: human approval is oversight, not containment, and unacceptable risk must propagate into a hard-stop verdict.",
        "",
        "This plot should be read as a prioritization aid for study and regression design, not as a quantitative course evaluation.",
    ])
    return "\n".join(lines).strip() + "\n"


def ascii_histogram(rows: list[dict[str, Any]], width: int = 34) -> list[str]:
    max_probability = max([row["probability"] for row in rows] + [0.0])
    output = []
    for row in rows:
        bar_len = int(round((row["probability"] / max_probability) * width)) if max_probability else 0
        output.append(f"W{row['week']:02d} {row['module_id']:<7} | {'#' * bar_len:<{width}} | {row['probability']:.3f}")
    return output


def top_weeks(analysis: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    return sorted(analysis["weeks"], key=lambda row: row["probability"], reverse=True)[:limit]


if __name__ == "__main__":
    raise SystemExit(main())

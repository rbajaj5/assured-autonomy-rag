from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare backend stability runs and compute paper-card RAG lift.")
    parser.add_argument("stability_json", nargs="+")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.stability_json]
    rows = compare_runs(runs)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, Path(args.output_csv))
    Path(args.output_md).write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    return 0


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (run.get("provider", ""), run.get("model", ""), run.get("fixture_name", ""))
        grouped[key][run.get("rag_mode", "none")] = run
    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        key = (run.get("provider", ""), run.get("model", ""), run.get("fixture_name", ""))
        pair = grouped.get(key, {})
        no_rag = pair.get("none", {}).get("metrics", {})
        with_rag = pair.get("paper_cards", {}).get("metrics", {})
        rag_lift = ""
        if no_rag and with_rag:
            rag_lift = round(
                float(no_rag.get("deterministic_disagreement_rate", 0.0))
                - float(with_rag.get("deterministic_disagreement_rate", 0.0)),
                4,
            )
        rows.append({
            "provider": run.get("provider", ""),
            "model": run.get("model", ""),
            "fixture": run.get("fixture_name", ""),
            "rag_mode": run.get("rag_mode", "none"),
            "parsed_successes": metrics.get("parsed_successes", 0),
            "deterministic_disagreement_rate": metrics.get("deterministic_disagreement_rate", 0.0),
            "approve_unacceptable_frequency": metrics.get("approve_unacceptable_frequency", 0.0),
            "label_variability_rate": metrics.get("label_variability_rate", 0.0),
            "evidence_citation_rate": metrics.get("evidence_citation_rate", 0.0),
            "unsupported_claim_rate": metrics.get("unsupported_claim_rate", 0.0),
            "rag_lift": rag_lift,
            "interpretation": run.get("interpretation", {}).get("label", ""),
        })
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "provider",
        "model",
        "fixture",
        "rag_mode",
        "parsed_successes",
        "deterministic_disagreement_rate",
        "approve_unacceptable_frequency",
        "label_variability_rate",
        "evidence_citation_rate",
        "unsupported_claim_rate",
        "rag_lift",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Backend Stability Comparison",
        "",
        "All backend outputs are advisory only. The deterministic assurance compiler remains authoritative.",
        "",
        "| Provider/model | Fixture | RAG mode | Parsed successes | Disagreement | Approve/unacceptable | Label variability | Evidence citation | RAG lift | Interpretation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                f"{row['provider']}/{row['model']}",
                row["fixture"],
                row["rag_mode"],
                str(row["parsed_successes"]),
                str(row["deterministic_disagreement_rate"]),
                str(row["approve_unacceptable_frequency"]),
                str(row["label_variability_rate"]),
                str(row["evidence_citation_rate"]),
                str(row["rag_lift"]),
                row["interpretation"],
            ])
            + " |"
        )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

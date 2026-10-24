"""Audit and render full-sweep paper cards into readable study notes."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "paper_central_claim",
    "method_or_framework",
    "specific_concepts",
    "capacity_allocation_relevance",
    "techtrack_upgrade_relevance",
    "assurance_tests_or_metrics",
    "assumptions_limits",
    "misuse_or_overclaim_warnings",
    "citation_snippets",
    "presentation_questions",
]


def curate_cards(card_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = out_dir / "readable_cards"
    readable_dir.mkdir(exist_ok=True)
    rows = []
    for path in sorted(card_dir.glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        audit = audit_card(card)
        rows.append(audit)
        out_path = readable_dir / (path.stem + ".md")
        out_path.write_text(render_readable_card(card, audit), encoding="utf-8")
    with (out_dir / "card_quality_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "short_path", "quality_score", "verdict", "issues", "char_count",
            "citation_count", "techtrack_specificity", "limit_count",
        ])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "card_quality_audit.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'card_quality_audit.csv'}")
    print(f"Wrote readable cards to {readable_dir}")
    return rows


def audit_card(card: dict) -> dict:
    text = "\n".join(_flatten(card.get(field)) for field in FIELDS)
    issues = []
    if card.get("status") != "complete":
        issues.append("not complete")
    if len(_flatten(card.get("paper_central_claim"))) < 350:
        issues.append("central claim too short")
    if _count_citations(text) < 6:
        issues.append("weak citation/page anchoring")
    if _specificity_score(text) < 8:
        issues.append("weak TechTrack/capacity specificity")
    if _limit_count(text) < 3:
        issues.append("few explicit limits/overclaim controls")
    if _generic_ratio(text) > 6:
        issues.append("generic phrasing may dominate")
    if "does not directly" not in text.lower() and "weak direct" not in text.lower() and "not evidence" not in text.lower():
        issues.append("may over-transfer paper into project")
    score = max(0, 100 - 12 * len(issues))
    verdict = "study-grade" if score >= 84 else "needs review" if score >= 64 else "repair"
    return {
        "short_path": card.get("short_path", ""),
        "quality_score": score,
        "verdict": verdict,
        "issues": "; ".join(issues),
        "char_count": len(text),
        "citation_count": _count_citations(text),
        "techtrack_specificity": _specificity_score(text),
        "limit_count": _limit_count(text),
    }


def render_readable_card(card: dict, audit: dict) -> str:
    lines = [
        f"# {card.get('short_path', 'Paper Card')}",
        "",
        f"Module: {card.get('module_id')} {card.get('module_name')}",
        f"Quality: {audit['verdict']} ({audit['quality_score']}/100)",
    ]
    if audit["issues"]:
        lines.append(f"Issues: {audit['issues']}")
    lines.append("")
    sections = [
        ("Central Claim", card.get("paper_central_claim")),
        ("Method / Framework", card.get("method_or_framework")),
        ("Key Concepts", card.get("specific_concepts")),
        ("Capacity Allocation Relevance", card.get("capacity_allocation_relevance")),
        ("TechTrack Upgrade Relevance", card.get("techtrack_upgrade_relevance")),
        ("Tests And Metrics", card.get("assurance_tests_or_metrics")),
        ("Assumptions And Limits", card.get("assumptions_limits")),
        ("Overclaim Warnings", card.get("misuse_or_overclaim_warnings")),
        ("Citation Anchors", card.get("citation_snippets")),
        ("Presentation Questions", card.get("presentation_questions")),
    ]
    for title, value in sections:
        lines.extend(["", f"## {title}", ""])
        lines.extend(_render_value(value))
    lines.append("")
    return "\n".join(lines)


def _render_value(value) -> list[str]:
    if value is None:
        return ["Not provided."]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(f"- {item}")
            else:
                flat = _flatten(item)
                out.append(f"- {flat}")
        return out or ["Not provided."]
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            label = str(key).replace("_", " ").title()
            if isinstance(item, list):
                out.append(f"**{label}:**")
                out.extend(f"- {_flatten(x)}" for x in item)
            else:
                out.append(f"**{label}:** {_flatten(item)}")
        return out or ["Not provided."]
    return [_flatten(value)]


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {_flatten(v)}" for k, v in value.items())
    return str(value)


def _count_citations(text: str) -> int:
    return len(re.findall(r"(?:M\d{2}|M01-02|EXT)[^\\s,;]*\\.pdf|p\\.\\s*\\d+|pp\\.\\s*\\d+", text))


def _specificity_score(text: str) -> int:
    terms = [
        "yolo", "cnn", "anchor", "small-object", "input resolution", "mAP",
        "calibration", "hard negative", "traffic-sign", "german", "spectral",
        "pca", "mahalanobis", "runtime", "adversarial", "shift", "latency",
        "verification", "ood", "confidence", "feature", "embedding",
    ]
    lower = text.lower()
    return sum(1 for term in terms if term.lower() in lower)


def _limit_count(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in [
        "do not", "does not", "weak", "limit", "overclaim", "not directly",
        "not evidence", "cannot", "should not",
    ])


def _generic_ratio(text: str) -> float:
    lower = text.lower()
    generic = sum(lower.count(term) for term in ["important", "relevant", "useful", "framework", "should", "could"])
    specific = max(1, _specificity_score(text))
    return generic / specific


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-dir", required=True)
    parser.add_argument("--out-dir", default="card_curation_runs")
    args = parser.parse_args(argv)
    curate_cards(Path(args.card_dir), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render compact, study-usable cards from SIRA paper-card JSON files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def curate_sira_cards(card_dir: Path, out_dir: Path, *, force: bool = False) -> dict:
    if not card_dir.exists():
        raise FileNotFoundError(card_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"card_dir": str(card_dir), "out_dir": str(out_dir), "cards": []}
    for path in sorted(card_dir.glob("*.json")):
        out = out_dir / (path.stem + ".readable.md")
        if out.exists() and not force:
            manifest["cards"].append({"source": str(path), "readable": str(out), "status": "exists"})
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        md = render_readable_card(card)
        out.write_text(md, encoding="utf-8")
        manifest["cards"].append({"source": str(path), "readable": str(out), "status": "written"})
    (out_dir / "readable_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote readable SIRA cards to {out_dir}")
    return manifest


def render_readable_card(card: dict) -> str:
    title = card.get("short_path") or card.get("filename") or "Paper"
    module = f"{card.get('module_id', '')} {card.get('module_name', '')}".strip()
    actual = card.get("paper_actual_contribution")
    tech = card.get("techtrack_upgrade_role")
    capacity = card.get("capacity_allocation_role")
    tests = card.get("assurance_test_role")
    limits = card.get("what_not_to_claim")
    notes = card.get("study_notes")
    anchors = card.get("citation_anchors")

    lines = [
        f"# {title}",
        "",
        f"Module: {module}",
        "",
        "## One-Sentence Takeaway",
        "",
        _sentence_takeaway(actual, tech),
        "",
        "## What The Paper Actually Does",
        "",
    ]
    lines.extend(_bullets(_extract_actual(actual), max_items=6))
    lines.extend(["", "## Evidence You Can Cite", ""])
    lines.extend(_bullets(_extract_anchors(anchors), max_items=8))
    lines.extend(["", "## How It Helps The TechTrack Upgrade", ""])
    lines.extend(_bullets(_extract_transfer(tech), max_items=6))
    lines.extend(["", "## Capacity / Geometry Relevance", ""])
    lines.extend(_bullets(_extract_capacity(capacity), max_items=4))
    lines.extend(["", "## Tests Or Metrics It Justifies", ""])
    lines.extend(_bullets(_extract_tests(tests), max_items=7))
    lines.extend(["", "## What Not To Claim", ""])
    lines.extend(_bullets(_extract_limits(limits), max_items=6))
    lines.extend(["", "## Coursework Use", ""])
    lines.extend(_bullets(_extract_notes(notes), max_items=5))
    lines.append("")
    return "\n".join(lines)


def _sentence_takeaway(actual, tech) -> str:
    summary = _find_first(actual, ["summary", "central_claim", "role"])
    if summary:
        return _clean(summary)
    items = _extract_actual(actual)
    if items:
        return _clean(items[0])
    items = _extract_transfer(tech)
    if items:
        return _clean(items[0])
    return "This card needs manual review; no clear takeaway was found."


def _extract_actual(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows = []
        for key in ("summary", "paper_actual", "actual_scope", "actual_claims", "specific_contributions", "page_anchored_points"):
            rows.extend(_items_from(value.get(key)))
        if not rows:
            rows.extend(_items_from(value))
        return rows
    return _items_from(value)


def _extract_transfer(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows = []
        boundary = value.get("actual_to_transfer_boundary")
        if isinstance(boundary, dict):
            transfer = boundary.get("transfer_to_techtrack")
            if transfer:
                rows.append(transfer)
        for key in ("strong_roles", "supported_transfers", "strong_transfer", "transfers_to_techtrack", "recommended_tests_for_techtrack"):
            rows.extend(_items_from(value.get(key)))
        if not rows:
            rows.extend(_items_from(value.get("role")))
            rows.extend(_items_from(value.get("how_it_helps")))
        return rows
    return _items_from(value)


def _extract_capacity(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows = []
        for key in ("role", "paper_support_level", "paper_actual", "supported_links", "techtrack_mapping", "transfer_to_techtrack"):
            rows.extend(_items_from(value.get(key)))
        return rows or _items_from(value)
    return _items_from(value)


def _extract_tests(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows = []
        for key in ("tests_supported_by_the_paper", "recommended_tests_for_techtrack", "paper_supported_assurance_elements", "test_categories_for_techtrack", "usable_as_meta_assurance"):
            rows.extend(_items_from(value.get(key)))
        return rows or _items_from(value)
    return _items_from(value)


def _extract_limits(value) -> list[str]:
    return _items_from(value)


def _extract_notes(value) -> list[str]:
    return _items_from(value)


def _extract_anchors(value) -> list[str]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                pages = _pages(item)
                anchor = item.get("anchor") or item.get("point") or item.get("claim") or item.get("description")
                rows.append(f"{_clean(anchor or key)}{pages}")
            else:
                rows.append(f"{_clean(key)}: {_clean(item)}")
    elif isinstance(value, list):
        for item in value:
            rows.append(_flatten_item(item))
    elif value:
        rows.append(str(value))
    return rows


def _items_from(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [_flatten_item(item) for item in value if item not in (None, "", [])]
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if key in {"raw_model_output", "status", "validation_error"}:
                continue
            if isinstance(item, (list, dict)):
                nested = _items_from(item)
                if nested:
                    rows.extend(nested)
            else:
                rows.append(f"{_label(key)}: {_clean(item)}")
        return rows
    return [str(value)]


def _flatten_item(item) -> str:
    if isinstance(item, str):
        return _clean(item)
    if isinstance(item, dict):
        pages = _pages(item)
        preferred = (
            item.get("claim")
            or item.get("point")
            or item.get("contribution")
            or item.get("role")
            or item.get("test")
            or item.get("element")
            or item.get("category")
            or item.get("note")
            or item.get("use")
            or item.get("techtrack_use")
            or item.get("upgrade_action")
            or item.get("description")
            or item.get("anchor")
            or item.get("overclaim")
            or item.get("claim_to_avoid")
        )
        if preferred:
            details = []
            for key in (
                "paper_actual_basis", "basis_in_paper", "paper_basis", "techtrack_transfer",
                "techtrack_translation", "implementation", "reason", "correction", "caveat",
                "support_level", "evidence_status",
            ):
                if item.get(key):
                    details.append(f"{_label(key)}: {_clean(item[key])}")
            detail_text = " " + " ".join(details) if details else ""
            return f"{_clean(preferred)}{pages}{detail_text}"
        return "; ".join(f"{_label(k)}: {_clean(v)}" for k, v in item.items() if k != "pages")
    if isinstance(item, list):
        return "; ".join(_flatten_item(x) for x in item)
    return _clean(item)


def _find_first(value, keys: list[str]) -> str:
    if isinstance(value, dict):
        for key in keys:
            if value.get(key):
                return str(value[key])
    if isinstance(value, str):
        return value
    return ""


def _pages(item: dict) -> str:
    pages = item.get("pages") or item.get("page")
    if pages is None:
        return ""
    if isinstance(pages, list):
        return f" (pp. {', '.join(str(p) for p in pages)})"
    return f" (p. {pages})"


def _bullets(items: list[str], max_items: int) -> list[str]:
    clean = []
    seen = set()
    for item in items:
        text = _clean(item)
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    if not clean:
        return ["- Needs review; no usable item found."]
    return [f"- {text}" for text in clean[:max_items]]


def _label(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


def _clean(value) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    curate_sira_cards(Path(args.card_dir), Path(args.out_dir), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

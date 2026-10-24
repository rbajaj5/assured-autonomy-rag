from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .blind_spot_atlas import BLIND_SPOTS, assign_support_level, retrieve_grounding
from .documents import ingest_paths
from .retrieval import LocalRetriever
from .schema import Chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether existing paper cards support Mistral blind spots.")
    parser.add_argument("--atlas", default="reports/mistral_blind_spot_atlas.json")
    parser.add_argument("--output-md", default="reports/mistral_paper_card_audit.md")
    parser.add_argument("--output-json", default="reports/mistral_paper_card_audit.json")
    parser.add_argument("--output-csv", default="reports/mistral_paper_card_mapping.csv")
    parser.add_argument("--citations-k", type=int, default=8)
    parser.add_argument("--no-embeddings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    atlas = json.loads(Path(args.atlas).read_text(encoding="utf-8"))
    paper_roots = locate_paper_card_roots()
    raw_roots = [Path("docs"), Path("readings"), Path("corpus")]
    audit = build_paper_card_audit(
        atlas,
        paper_roots=paper_roots,
        raw_roots=raw_roots,
        citations_k=args.citations_k,
        use_embeddings=not args.no_embeddings,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_mapping_csv(audit, Path(args.output_csv))
    Path(args.output_md).write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    return 0


def locate_paper_card_roots(base: Path = Path(".")) -> list[Path]:
    roots = []
    for path in base.rglob("*"):
        if not path.is_dir():
            continue
        if path.name in {"paper_cards", "readable_cards"}:
            roots.append(path)
    return sorted(roots)


def build_paper_card_audit(
    atlas: dict[str, Any],
    *,
    paper_roots: list[Path],
    raw_roots: list[Path],
    citations_k: int,
    use_embeddings: bool,
) -> dict[str, Any]:
    paper_chunks = ingest_paths(paper_roots)
    raw_chunks = ingest_paths(raw_roots)
    paper_retriever = LocalRetriever(paper_chunks, use_embeddings=use_embeddings)
    raw_retriever = LocalRetriever(raw_chunks, use_embeddings=use_embeddings)
    atlas_by_id = {item["blind_spot_id"]: item for item in atlas.get("blind_spots", [])}
    mappings = []
    for blind_spot_id, spec in BLIND_SPOTS.items():
        paper_hits = retrieve_grounding(spec, retriever=paper_retriever, k=citations_k)
        raw_hits = retrieve_grounding(spec, retriever=raw_retriever, k=citations_k)
        paper_support = assign_support_level(paper_hits)
        raw_support = assign_support_level(raw_hits)
        mappings.append({
            "blind_spot_id": blind_spot_id,
            "name": spec["name"],
            "atlas_support_level": atlas_by_id.get(blind_spot_id, {}).get("reading_list_support_level", "missing"),
            "paper_card_support_level": paper_support,
            "raw_support_level": raw_support,
            "paper_card_ids": [_card_id(hit) for hit in paper_hits],
            "paper_card_excerpts": [
                {
                    "paper_card_id": _card_id(hit),
                    "filename": hit["filename"],
                    "chunk_id": hit["chunk_id"],
                    "excerpt": hit["excerpt"],
                    "relevance": hit["relevance"],
                    "concept_supported": hit["concept_supported"],
                }
                for hit in paper_hits
            ],
            "raw_fallback_excerpts": [
                {
                    "filename": hit["filename"],
                    "chunk_id": hit["chunk_id"],
                    "excerpt": hit["excerpt"],
                    "relevance": hit["relevance"],
                }
                for hit in raw_hits
            ],
            "diagnosis": diagnose_retrieval(blind_spot_id, paper_hits, raw_hits, len(paper_chunks)),
            "recommend_new_cards_only_if": recommend_new_cards_only_if(blind_spot_id, paper_support),
        })
    return {
        "paper_card_roots": [str(path) for path in paper_roots],
        "paper_card_chunk_count": len(paper_chunks),
        "raw_chunk_count": len(raw_chunks),
        "mappings": mappings,
        "strong_paper_card_support": [m["blind_spot_id"] for m in mappings if m["paper_card_support_level"] == "strong"],
        "failed_paper_card_support": [m["blind_spot_id"] for m in mappings if m["paper_card_support_level"] in {"weak", "missing"}],
    }


def write_mapping_csv(audit: dict[str, Any], path: Path) -> None:
    fields = [
        "blind_spot_id",
        "name",
        "paper_card_support_level",
        "raw_support_level",
        "paper_card_ids",
        "diagnosis",
        "recommend_new_cards_only_if",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mapping in audit["mappings"]:
            writer.writerow({
                "blind_spot_id": mapping["blind_spot_id"],
                "name": mapping["name"],
                "paper_card_support_level": mapping["paper_card_support_level"],
                "raw_support_level": mapping["raw_support_level"],
                "paper_card_ids": ";".join(mapping["paper_card_ids"]),
                "diagnosis": mapping["diagnosis"],
                "recommend_new_cards_only_if": mapping["recommend_new_cards_only_if"],
            })


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Mistral Paper-Card Retrieval Audit",
        "",
        "This audit checks whether existing paper cards are retrieved for the Mistral blind spots before falling back to raw readings. No new paper cards were created.",
        "",
        "## Located Paper-Card Roots",
        "",
    ]
    for root in audit["paper_card_roots"]:
        lines.append(f"- `{root}`")
    lines.extend([
        "",
        f"- Paper-card chunks indexed: {audit['paper_card_chunk_count']}",
        f"- Raw fallback chunks indexed: {audit['raw_chunk_count']}",
        "",
        "## Summary",
        "",
        f"- Blind spots with strong paper-card support: {', '.join(f'`{item}`' for item in audit['strong_paper_card_support']) or 'none'}",
        f"- Blind spots failing to retrieve relevant paper cards: {', '.join(f'`{item}`' for item in audit['failed_paper_card_support']) or 'none'}",
        "",
        "## Mapping",
        "",
        "| Blind spot | Paper-card support | Raw support | Paper cards | Diagnosis |",
        "|---|---|---|---|---|",
    ])
    for mapping in audit["mappings"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{mapping['blind_spot_id']}`",
                mapping["paper_card_support_level"],
                mapping["raw_support_level"],
                ", ".join(f"`{item}`" for item in mapping["paper_card_ids"][:5]) or "none",
                mapping["diagnosis"],
            ])
            + " |"
        )
    lines.extend(["", "## Per-Blind-Spot Detail", ""])
    for mapping in audit["mappings"]:
        lines.extend([
            f"### {mapping['name']}",
            "",
            f"- Blind spot: `{mapping['blind_spot_id']}`",
            f"- Paper-card support: {mapping['paper_card_support_level']}",
            f"- Raw fallback support: {mapping['raw_support_level']}",
            f"- Diagnosis: {mapping['diagnosis']}",
            f"- New-card recommendation rule: {mapping['recommend_new_cards_only_if']}",
            "",
            "Relevant paper-card excerpts:",
        ])
        if mapping["paper_card_excerpts"]:
            for item in mapping["paper_card_excerpts"][:5]:
                lines.append(f"- `{item['paper_card_id']}` ({item['relevance']}): {item['excerpt']}")
        else:
            lines.append("- No relevant paper-card excerpt retrieved.")
        lines.extend(["", "Raw fallback excerpts:"])
        if mapping["raw_fallback_excerpts"]:
            for item in mapping["raw_fallback_excerpts"][:3]:
                lines.append(f"- `{item['filename']}#{item['chunk_id']}` ({item['relevance']}): {item['excerpt']}")
        else:
            lines.append("- No raw fallback excerpt retrieved.")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def diagnose_retrieval(blind_spot_id: str, paper_hits: list[dict[str, Any]], raw_hits: list[dict[str, Any]], paper_chunk_count: int) -> str:
    if paper_chunk_count == 0:
        return "No paper-card chunks were indexed; likely path/discovery issue."
    support = assign_support_level(paper_hits)
    if support == "strong":
        return "Paper cards retrieve strongly; atlas should use the card tier before raw readings."
    if support == "partial":
        return "Paper cards retrieve partially; query phrasing or card chunk granularity may be limiting exact concept match."
    if not paper_hits and raw_hits:
        return "Raw readings retrieve but paper cards do not; likely paper-card naming/metadata or query phrasing gap."
    if paper_hits and assign_support_level(raw_hits) in {"strong", "partial"}:
        return "Paper cards retrieve weakly while raw text is stronger; likely card summarization omitted the assurance concept."
    if paper_hits:
        return "Paper cards retrieve weakly; likely concept wording mismatch, overly broad chunks, or cards biased toward other project hooks."
    return "Neither paper cards nor raw fallback retrieved well; likely corpus gap or query vocabulary mismatch."


def recommend_new_cards_only_if(blind_spot_id: str, support: str) -> str:
    if support in {"strong", "partial"}:
        return "Do not create new cards yet; improve retrieval/linking to existing cards first."
    return "Recommend new or repaired cards only if manual review confirms no existing card covers this concept."


def _card_id(hit: dict[str, Any]) -> str:
    path = Path(hit.get("filename", ""))
    return path.stem


if __name__ == "__main__":
    raise SystemExit(main())

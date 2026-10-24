"""Page-level evidence search across course and external PDFs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import AA_MODULES, module_id_from_dirname
from .pdf_text import extract_pdf_pages


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "use", "using", "with", "we", "what", "when", "where",
    "which", "why",
}


@dataclass
class DeepSearchConfig:
    papers_dirs: list[Path]
    query: str
    out_dir: Path = Path("deep_evidence_runs")
    top_pages: int = 30
    context_chars: int = 900
    max_pages_per_pdf: int = 500
    include_external: bool = True


def run_deep_search(cfg: DeepSearchConfig) -> dict:
    run_dir = _new_run_dir(cfg.out_dir)
    terms = _query_terms(cfg.query)
    pdfs = _discover_pdfs(cfg.papers_dirs)
    pages = []
    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] scan {pdf['short_path']}")
        for page_no, text in enumerate(
            extract_pdf_pages(pdf["path"], max_pages=cfg.max_pages_per_pdf),
            start=1,
        ):
            norm = _normalize(text)
            if not norm:
                continue
            score, hits = _score_page(norm, terms)
            if score <= 0:
                continue
            pages.append({
                **pdf,
                "page": page_no,
                "score": score,
                "hits": hits,
                "snippet": _snippet(text, hits, cfg.context_chars),
            })
    ranked = sorted(pages, key=lambda r: r["score"], reverse=True)[: cfg.top_pages]
    result = {
        "query": cfg.query,
        "terms": terms,
        "papers_dirs": [str(p) for p in cfg.papers_dirs],
        "page_count_scored": len(pages),
        "top_pages": _jsonable_rows(ranked),
    }
    (run_dir / "deep_pdf_evidence.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (run_dir / "deep_pdf_evidence.md").write_text(
        _render_markdown(result),
        encoding="utf-8",
    )
    print(f"Saved deep evidence directory: {run_dir}")
    return result


def _discover_pdfs(papers_dirs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for root in papers_dirs:
        root = root.resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pdf")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            module_dir = next((p for p in path.parents if module_id_from_dirname(p.name)), None)
            module_id = module_id_from_dirname(module_dir.name) if module_dir else "EXT"
            module_name = AA_MODULES.get(module_id or "", module_dir.name if module_dir else "External Reading")
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path.name
            rows.append({
                "path": path,
                "short_path": str(rel).replace("\\", "/"),
                "module_id": module_id,
                "module_name": module_name,
                "filename": path.name,
            })
    return rows


def _query_terms(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
    terms = []
    for word in words:
        if word not in STOPWORDS and word not in terms:
            terms.append(word)
    phrases = re.findall(r'"([^"]{4,})"', query.lower())
    for phrase in phrases:
        clean = " ".join(re.findall(r"[a-z0-9_-]+", phrase))
        if clean and clean not in terms:
            terms.insert(0, clean)
    return terms


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _score_page(text: str, terms: list[str]) -> tuple[float, list[str]]:
    hits = []
    score = 0.0
    for term in terms:
        if " " in term:
            count = text.count(term)
        else:
            count = len(re.findall(rf"\b{re.escape(term)}s?\b", text))
        if count:
            hits.append(term)
            score += (2.5 if " " in term else 1.0) * (1.0 + math.log(count))
    # Reward dense conceptual overlap but dampen single-term spam.
    score *= 1.0 + min(len(hits), 8) / 5.0
    return score, hits


def _snippet(text: str, hits: list[str], context_chars: int) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    lower = flat.lower()
    starts = [lower.find(hit) for hit in hits if lower.find(hit) >= 0]
    if not starts:
        return flat[:context_chars]
    center = min(starts)
    start = max(0, center - context_chars // 3)
    end = min(len(flat), start + context_chars)
    return flat[start:end].strip()


def _jsonable_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        copy = dict(row)
        copy["path"] = str(copy["path"])
        out.append(copy)
    return out


def _render_markdown(result: dict) -> str:
    lines = [
        "# Deep PDF Evidence",
        "",
        f"Query: {result['query']}",
        "",
        f"Terms: {', '.join(result['terms'])}",
        "",
    ]
    for i, row in enumerate(result["top_pages"], start=1):
        lines.extend([
            f"## {i}. {row['short_path']} p. {row['page']}",
            "",
            f"Module: {row['module_id']} {row['module_name']}",
            "",
            f"Score: {row['score']:.2f}",
            "",
            f"Hits: {', '.join(row['hits'])}",
            "",
            row["snippet"],
            "",
            f"Local path: {row['path']}",
            "",
        ])
    return "\n".join(lines)


def _new_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = base_dir / f"deep_pdf_search_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", action="append",
                        default=[],
                        help="Directory to scan. Repeat for course and external papers.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--out-dir", default="deep_evidence_runs")
    parser.add_argument("--top-pages", type=int, default=30)
    parser.add_argument("--context-chars", type=int, default=900)
    parser.add_argument("--max-pages-per-pdf", type=int, default=500)
    args = parser.parse_args(argv)
    dirs = [Path(p) for p in args.papers_dir] or [
        Path('papers/papers'),
        Path('papers/external'),
    ]
    run_deep_search(
        DeepSearchConfig(
            papers_dirs=dirs,
            query=args.query,
            out_dir=Path(args.out_dir),
            top_pages=args.top_pages,
            context_chars=args.context_chars,
            max_pages_per_pdf=args.max_pages_per_pdf,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

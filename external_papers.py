"""Download and register external reading candidates for semantic digestion."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


def fetch_external_papers(
    candidates_path: str | Path,
    out_dir: str | Path,
    *,
    only: list[str] | None = None,
    timeout: int = 60,
    overwrite: bool = False,
) -> dict:
    candidates_file = Path(candidates_path)
    target_root = Path(out_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
    wanted = set(only or [])
    rows = []
    for item in candidates:
        if wanted and item.get("id") not in wanted:
            continue
        row = _fetch_one(item, target_root, timeout=timeout, overwrite=overwrite)
        rows.append(row)
        status = row["status"]
        detail = row.get("path") or row.get("reason") or row.get("pdf_url")
        print(f"[{status}] {item.get('id')}: {detail}")
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "candidates_path": str(candidates_file),
        "out_dir": str(target_root),
        "items": rows,
    }
    manifest_path = target_root / "external_papers_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return manifest


def _fetch_one(item: dict, target_root: Path, *, timeout: int, overwrite: bool) -> dict:
    pdf_url = _candidate_pdf_url(str(item.get("url", "")))
    if not pdf_url:
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "status": "skipped",
            "reason": "no direct PDF URL could be inferred; download manually if needed",
        }
    module = _primary_module(item)
    folder = target_root / f"{module}_ExternalReadings"
    folder.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(f"{item.get('year', 'n.d.')}-{item.get('id', 'external')}.pdf")
    path = folder / filename
    if path.exists() and not overwrite:
        return _row(item, pdf_url, path, "exists")
    headers = {"User-Agent": "AATE academic reading intake/1.0"}
    try:
        resp = requests.get(pdf_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "pdf_url": pdf_url,
            "status": "failed",
            "reason": str(exc),
        }
    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not resp.content.startswith(b"%PDF"):
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "pdf_url": pdf_url,
            "status": "failed",
            "reason": f"response did not look like a PDF: {content_type or 'unknown content-type'}",
        }
    path.write_bytes(resp.content)
    sidecar = path.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(item | {"pdf_url": pdf_url}, indent=2), encoding="utf-8")
    return _row(item, pdf_url, path, "downloaded")


def _row(item: dict, pdf_url: str, path: Path, status: str) -> dict:
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "authors": item.get("authors"),
        "year": item.get("year"),
        "url": item.get("url"),
        "pdf_url": pdf_url,
        "modules": item.get("modules", []),
        "status": status,
        "path": str(path),
        "semantic_sweep_path_hint": str(path.parent.parent),
    }


def _candidate_pdf_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.endswith("arxiv.org"):
        match = re.search(r"/abs/([^/?#]+)", parsed.path)
        if match:
            return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
        if "/pdf/" in parsed.path:
            return url
    if parsed.path.lower().endswith(".pdf"):
        return url
    return None


def _primary_module(item: dict) -> str:
    modules = item.get("modules") or ["M07"]
    first = str(modules[0]).strip() or "M07"
    return first if re.fullmatch(r"M\d{2}|M01-02", first) else "M07"


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^\w.\-]+", "-", value, flags=re.ASCII).strip("-")
    return value or "external-paper.pdf"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/external_reading_candidates.json")
    parser.add_argument("--out-dir", default='papers/external')
    parser.add_argument("--only", nargs="*", help="Optional candidate ids to fetch")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    fetch_external_papers(
        args.candidates,
        args.out_dir,
        only=args.only,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

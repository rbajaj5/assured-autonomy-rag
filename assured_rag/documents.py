from __future__ import annotations

import re
from pathlib import Path

from .schema import Chunk


SUPPORTED_EXTS = {".md", ".txt", ".pdf"}


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _read_pdf(path)
    return ""


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore

        parts: list[str] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    parts.append(f"\n\n[Page {i + 1}]\n{text}")
        return "\n".join(parts)
    except Exception:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"\n\n[Page {i + 1}]\n{text}")
            return "\n".join(parts)
        except Exception:
            return ""


def iter_source_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    return sorted(files)


def chunk_text(
    text: str,
    *,
    filename: str,
    chunk_chars: int = 1800,
    overlap: int = 220,
) -> list[Chunk]:
    text = text.replace("\r\n", "\n")
    headings = _heading_spans(text)
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start + chunk_chars // 2, end), text.rfind(". ", start + chunk_chars // 2, end))
            if boundary > start:
                end = boundary + 1
        raw = text[start:end].strip()
        if raw:
            idx += 1
            chunks.append(
                Chunk(
                    filename=filename,
                    chunk_id=f"chunk-{idx:04d}",
                    text=raw,
                    start_char=start,
                    end_char=end,
                    section=_section_for_offset(headings, start),
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest_paths(paths: list[Path], *, chunk_chars: int = 1800, overlap: int = 220) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_source_files(paths):
        text = read_document(path)
        if not text.strip():
            continue
        chunks.extend(
            chunk_text(
                text,
                filename=str(path),
                chunk_chars=chunk_chars,
                overlap=overlap,
            )
        )
    return chunks


def _heading_spans(text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    for match in re.finditer(r"(?m)^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,:;()/&-]{8,80})\s*$", text):
        title = match.group(1).strip("# ").strip()
        if title:
            spans.append((match.start(), title[:120]))
    return spans


def _section_for_offset(headings: list[tuple[int, str]], offset: int) -> str:
    current = ""
    for pos, title in headings:
        if pos <= offset:
            current = title
        else:
            break
    return current

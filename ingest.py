"""Corpus ingestion.

Walks a `papers/` directory of the form

    papers/
      M03_SafetySecurityPrivacy/foo.pdf
      M11_AdversarialAutonomy/bar.pdf
      ...

extracts page-level text, and emits page-anchored chunks with metadata.
Also accepts plain `.txt` and `.md` files inside module folders, which is
useful for synthetic test corpora.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from .config import IngestConfig, module_id_from_dirname, AA_MODULES


@dataclass
class Chunk:
    chunk_id: str          # stable id: <module>::<filename>::<page>::<offset>
    module_id: str         # e.g. "M03"
    module_name: str       # e.g. "Safety, Security, and Privacy"
    filename: str          # basename
    relpath: str           # path relative to corpus root
    page: int              # 1-indexed page number; 0 for non-PDF
    offset: int            # character offset within page
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        """Short, stable citation string used in prompts and outputs."""
        if self.page > 0:
            return f"[{self.module_id} / {self.filename}, p. {self.page}]"
        return f"[{self.module_id} / {self.filename}]"


def _extract_pdf_pages(path: Path, max_pages: int) -> list[str]:
    """Return list of page texts. Empty list on failure."""
    try:
        from .pdf_text import extract_pdf_pages
    except ImportError as e:
        raise RuntimeError("PDF extraction helpers unavailable.") from e
    try:
        return extract_pdf_pages(path, max_pages=max_pages)
    except Exception as e:
        print(f"  ! could not extract {path.name}: {e}")
        return []


def _normalize(text: str) -> str:
    # Collapse hyphenated line breaks (common artifact of PDF extraction)
    text = re.sub(r"-\n(?=\w)", "", text)
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, cfg: IngestConfig) -> Iterator[tuple[int, str]]:
    """Yield (offset, chunk_text). Sliding-window character chunking with overlap.

    Tries to break at paragraph or sentence boundaries near the cut point.
    """
    if len(text) < cfg.min_chunk_chars:
        if text.strip():
            yield 0, text
        return
    n = len(text)
    i = 0
    step = cfg.chunk_chars - cfg.chunk_overlap_chars
    while i < n:
        end = min(i + cfg.chunk_chars, n)
        if end < n:
            # Look for a clean break in the last 200 chars
            window = text[max(end - 200, i):end]
            for sep in ("\n\n", ". ", "\n", "; "):
                idx = window.rfind(sep)
                if idx >= 0:
                    end = max(end - 200, i) + idx + len(sep)
                    break
        chunk = text[i:end].strip()
        if len(chunk) >= cfg.min_chunk_chars or i == 0:
            yield i, chunk
        if end >= n:
            break
        i = max(end - cfg.chunk_overlap_chars, i + 1)


def ingest_course_corpus(corpus_dir: str | os.PathLike,
                         cfg: IngestConfig | None = None,
                         verbose: bool = True) -> list[Chunk]:
    """Walk `corpus_dir`, extract text, return a list of Chunk objects.

    Skips files outside known module subdirectories (with a warning).
    """
    cfg = cfg or IngestConfig()
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus dir not found: {root}")

    module_dirs = [
        p for p in root.rglob("*")
        if p.is_dir() and module_id_from_dirname(p.name) is not None
    ]
    if module_id_from_dirname(root.name) is not None:
        module_dirs.append(root)
    module_dirs = sorted(set(module_dirs))

    chunks: list[Chunk] = []
    for module_dir in module_dirs:
        module_id = module_id_from_dirname(module_dir.name)
        if module_id is None:
            if verbose:
                print(f"  [skip] {module_dir.name}: not a known AA module")
            continue
        module_name = AA_MODULES[module_id]
        if verbose:
            print(f"[{module_id}] {module_name}: {module_dir.name}")
        for fpath in sorted(module_dir.rglob("*")):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            relpath = str(fpath.relative_to(root))
            if ext == ".pdf":
                pages = _extract_pdf_pages(fpath, cfg.max_pages_per_doc)
                for page_idx, page_text in enumerate(pages, start=1):
                    norm = _normalize(page_text)
                    for offset, chunk in _chunk_text(norm, cfg):
                        cid = f"{module_id}::{fpath.name}::{page_idx:04d}::{offset:06d}"
                        chunks.append(Chunk(
                            chunk_id=cid,
                            module_id=module_id,
                            module_name=module_name,
                            filename=fpath.name,
                            relpath=relpath,
                            page=page_idx,
                            offset=offset,
                            text=chunk,
                        ))
            elif ext in (".txt", ".md"):
                try:
                    raw = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    if verbose:
                        print(f"  ! could not read {fpath.name}: {e}")
                    continue
                norm = _normalize(raw)
                for offset, chunk in _chunk_text(norm, cfg):
                    cid = f"{module_id}::{fpath.name}::0000::{offset:06d}"
                    chunks.append(Chunk(
                        chunk_id=cid,
                        module_id=module_id,
                        module_name=module_name,
                        filename=fpath.name,
                        relpath=relpath,
                        page=0,
                        offset=offset,
                        text=chunk,
                    ))
            # silently skip other extensions
    if verbose:
        n_files = len({(c.module_id, c.filename) for c in chunks})
        print(f"\nIngested {len(chunks)} chunks from {n_files} files.")
    return chunks

"""High-fidelity document text extraction helpers."""
from __future__ import annotations

from pathlib import Path


def extract_pdf_pages(path: str | Path, max_pages: int = 200) -> list[str]:
    """Extract per-page text, preferring PyMuPDF and falling back to pypdf."""
    pdf_path = Path(path)
    try:
        import fitz  # PyMuPDF

        pages: list[str] = []
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                pages.append(page.get_text("text") or "")
        return pages
    except ImportError:
        pass
    except Exception as exc:
        print(f"  ! PyMuPDF extraction failed for {pdf_path.name}: {exc}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF or pypdf for PDF extraction.") from exc

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        pages.append(page.extract_text() or "")
    return pages

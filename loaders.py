"""Load new cybersecurity items: PDF, URL, or pasted text."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class NewItem:
    text: str
    source_kind: str          # "pdf" | "url" | "text"
    source: str               # path, URL, or "<pasted text>"
    title: Optional[str] = None

    def header(self) -> str:
        bits = [f"Source kind: {self.source_kind}", f"Source: {self.source}"]
        if self.title:
            bits.append(f"Title: {self.title}")
        return "\n".join(bits)


# ---------------------------------------------------------------------------
def _load_pdf(path: str | Path) -> NewItem:
    from pypdf import PdfReader
    p = Path(path)
    reader = PdfReader(str(p))
    pieces: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pieces.append(page.extract_text() or "")
        except Exception:
            continue
    text = _normalize("\n\n".join(pieces))
    return NewItem(text=text, source_kind="pdf", source=str(p), title=p.name)


def _load_pdf_bytes(data: bytes, *, source: str, title: str | None = None) -> NewItem:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pieces: list[str] = []
    for page in reader.pages:
        try:
            pieces.append(page.extract_text() or "")
        except Exception:
            continue
    text = _normalize("\n\n".join(pieces))
    return NewItem(text=text, source_kind="pdf", source=source, title=title or source.rsplit("/", 1)[-1])


def _load_url(url: str, *, timeout: int = 20) -> NewItem:
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf"):
        return _load_pdf_bytes(resp.content, source=url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip clear boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header",
                      "aside", "form", "noscript"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else None)

    # Heuristic main-content extraction: prefer <article>, then <main>, then body
    container = soup.find("article") or soup.find("main") or soup.body or soup
    text = container.get_text(separator="\n", strip=True)
    text = _normalize(text)
    return NewItem(text=text, source_kind="url", source=url, title=title)


def _load_text(text: str) -> NewItem:
    return NewItem(text=_normalize(text), source_kind="text", source="<pasted text>")


def _normalize(text: str) -> str:
    text = re.sub(r"-\n(?=\w)", "", text)        # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
def load_new_item(input_value: str | Path) -> NewItem:
    """Dispatch a single string/path argument to the right loader.

    Heuristics:
      - starts with http:// or https://  → URL
      - filesystem path ending in .pdf   → PDF file
      - everything else                  → treat as pasted text
    """
    s = str(input_value).strip()
    if s.startswith("http://") or s.startswith("https://"):
        return _load_url(s)
    # Treat as a candidate path only if it's plausibly one. Long pasted
    # text would otherwise raise OSError("File name too long") from os.stat.
    if "\n" not in s and len(s) < 1024:
        try:
            p = Path(s)
            if p.exists():
                ext = p.suffix.lower()
                if ext == ".pdf":
                    return _load_pdf(p)
                if ext in (".txt", ".md") and p.stat().st_size < 5_000_000:
                    return _load_text(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    return _load_text(s)

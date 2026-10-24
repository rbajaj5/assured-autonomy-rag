from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .documents import ingest_paths
from .schema import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    method: str

    def citation(self) -> dict[str, Any]:
        row = self.chunk.citation()
        row["score"] = round(float(self.score), 4)
        row["method"] = self.method
        return row


class LocalRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ) -> None:
        self.chunks = chunks
        self.embedding_model_name = embedding_model
        self._model: Any | None = None
        self._embeddings: np.ndarray | None = None
        self._faiss_index: Any | None = None
        self._doc_freq = self._build_doc_freq(chunks)
        if use_embeddings and chunks:
            self._try_build_vector_index(embedding_model)

    @classmethod
    def from_default_dirs(
        cls,
        *,
        roots: list[Path] | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ) -> "LocalRetriever":
        roots = roots or [Path("docs"), Path("readings"), Path("corpus")]
        chunks = ingest_paths(roots)
        return cls(chunks, embedding_model=embedding_model, use_embeddings=use_embeddings)

    def search(self, query: str, *, k: int = 6) -> list[RetrievedChunk]:
        vector_hits = self._vector_search(query, k=k) if self._faiss_index is not None else []
        keyword_hits = self._keyword_search(query, k=k)
        merged: dict[tuple[str, str], RetrievedChunk] = {}
        for hit in vector_hits + keyword_hits:
            key = (hit.chunk.filename, hit.chunk.chunk_id)
            if key not in merged or hit.score > merged[key].score:
                merged[key] = hit
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]

    def _try_build_vector_index(self, embedding_model: str) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(embedding_model)
            texts = [chunk.text for chunk in self.chunks]
            embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._embeddings = np.asarray(embs, dtype="float32")
            index = faiss.IndexFlatIP(self._embeddings.shape[1])
            index.add(self._embeddings)
            self._faiss_index = index
        except Exception:
            self._model = None
            self._embeddings = None
            self._faiss_index = None

    def _vector_search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        if self._model is None or self._faiss_index is None:
            return []
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores, indices = self._faiss_index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        hits: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(RetrievedChunk(self.chunks[int(idx)], float(score), "vector"))
        return hits

    def _keyword_search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        q_terms = _tokens(query)
        if not q_terms:
            return []
        hits: list[RetrievedChunk] = []
        total = max(len(self.chunks), 1)
        for chunk in self.chunks:
            terms = _tokens(chunk.text)
            common = q_terms & terms
            if not common:
                continue
            score = 0.0
            for term in common:
                idf = math.log((1 + total) / (1 + self._doc_freq.get(term, 0))) + 1.0
                score += idf * (1.0 + min(chunk.text.lower().count(term), 5) / 5.0)
            hits.append(RetrievedChunk(chunk, score, "keyword"))
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    @staticmethod
    def _build_doc_freq(chunks: list[Chunk]) -> dict[str, int]:
        df: dict[str, int] = {}
        for chunk in chunks:
            for term in _tokens(chunk.text):
                df[term] = df.get(term, 0) + 1
        return df


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}

"""FAISS index over Chunk embeddings, with save/load and metadata-aware retrieval."""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import IndexConfig, RetrievalConfig
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class RagIndex:
    """Cosine-similarity FAISS index. Stores chunks alongside embeddings.

    Uses IndexFlatIP with L2-normalized vectors → equivalent to cosine sim.
    """

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray,
                 cfg: IndexConfig | None = None):
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise RuntimeError("faiss not installed. `pip install faiss-cpu`") from e
        self.cfg = cfg or IndexConfig()
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32, copy=False)
        if self.cfg.normalize:
            faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self._encoder = None  # lazily loaded

    # ------------------------------------------------------------------
    def _encoder_model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.cfg.embedding_model)
        return self._encoder

    def query(self, text: str, *, k: int = 8,
              min_score: float = 0.0,
              modules: Iterable[str] | None = None) -> list[Hit]:
        """Retrieve top-k most similar chunks. Optional module filter."""
        import faiss  # type: ignore
        enc = self._encoder_model()
        q = enc.encode([text], convert_to_numpy=True).astype(np.float32)
        if self.cfg.normalize:
            faiss.normalize_L2(q)
        # over-fetch a bit when filtering
        fetch = k * 4 if modules else k
        scores, idxs = self.index.search(q, min(fetch, len(self.chunks)))
        hits: list[Hit] = []
        wanted = set(modules) if modules else None
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            ch = self.chunks[i]
            if wanted is not None and ch.module_id not in wanted:
                continue
            if float(s) < min_score:
                continue
            hits.append(Hit(chunk=ch, score=float(s)))
            if len(hits) >= k:
                break
        return hits

    # ------------------------------------------------------------------
    def save(self, dest_dir: str | Path) -> None:
        """Persist embeddings, metadata, and config to disk."""
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        with (d / "chunks.pkl").open("wb") as f:
            pickle.dump(self.chunks, f)
        (d / "config.json").write_text(json.dumps({
            "embedding_model": self.cfg.embedding_model,
            "embedding_dim": int(self.embeddings.shape[1]),
            "normalize": self.cfg.normalize,
            "n_chunks": len(self.chunks),
        }, indent=2))

    @classmethod
    def load(cls, src_dir: str | Path) -> "RagIndex":
        d = Path(src_dir)
        emb = np.load(d / "embeddings.npy")
        with (d / "chunks.pkl").open("rb") as f:
            chunks = pickle.load(f)
        meta = json.loads((d / "config.json").read_text())
        cfg = IndexConfig(
            embedding_model=meta["embedding_model"],
            embedding_dim=meta["embedding_dim"],
            normalize=meta["normalize"],
        )
        # We saved normalized embeddings; tell __init__ not to renormalize.
        cfg_load = IndexConfig(**{**cfg.__dict__, "normalize": False})
        idx = cls(chunks, emb, cfg=cfg_load)
        idx.cfg = cfg  # restore original cfg for queries
        return idx


def build_index(chunks: list[Chunk], cfg: IndexConfig | None = None,
                verbose: bool = True) -> RagIndex:
    """Embed chunks with sentence-transformers and return a RagIndex."""
    cfg = cfg or IndexConfig()
    if not chunks:
        raise ValueError("No chunks to index.")
    from sentence_transformers import SentenceTransformer
    if verbose:
        print(f"Loading embedding model: {cfg.embedding_model}")
    model = SentenceTransformer(cfg.embedding_model)
    texts = [c.text for c in chunks]
    if verbose:
        print(f"Encoding {len(texts)} chunks...")
    emb = model.encode(texts, batch_size=32, show_progress_bar=verbose,
                       convert_to_numpy=True)
    return RagIndex(chunks, emb, cfg=cfg)


def format_hits_for_prompt(hits: list[Hit], max_chars_per_hit: int = 800) -> str:
    """Render retrieved passages with stable citation handles for the LLM."""
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        cite = h.chunk.citation()
        body = h.chunk.text
        if len(body) > max_chars_per_hit:
            body = body[:max_chars_per_hit].rstrip() + " […]"
        blocks.append(f"[R{i}] {cite}  (score={h.score:.3f})\n{body}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"

"""Hybrid vector + keyword retrieval with optional reranking."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import RetrievalConfig
from .indexing import Hit, RagIndex
from .syllabus import SyllabusNavigator


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


@dataclass
class HybridRetrievalConfig:
    top_k: int = 8
    vector_k: int = 24
    keyword_k: int = 24
    vector_weight: float = 0.65
    keyword_weight: float = 0.25
    required_weight: float = 0.10
    reranker_model: str | None = "BAAI/bge-reranker-base"
    prefer_required: bool = True


class HybridPaperRetriever:
    """Paper_Retriever tool: FAISS vector retrieval plus local BM25-style search."""

    def __init__(self, index: RagIndex, *, corpus_dir: str | Path,
                 cfg: HybridRetrievalConfig | None = None):
        self.index = index
        self.corpus_dir = Path(corpus_dir)
        self.cfg = cfg or HybridRetrievalConfig()
        self.navigator = SyllabusNavigator()
        self._doc_freq = self._build_doc_freq()
        self._reranker = None

    def _build_doc_freq(self) -> Counter[str]:
        df: Counter[str] = Counter()
        for chunk in self.index.chunks:
            df.update(set(_tokens(chunk.text)))
        return df

    def _keyword_scores(self, query: str, modules: Iterable[str] | None) -> dict[str, float]:
        q_terms = Counter(_tokens(query))
        wanted = set(modules) if modules else None
        n_docs = max(len(self.index.chunks), 1)
        scores: dict[str, float] = defaultdict(float)
        for chunk in self.index.chunks:
            if wanted and chunk.module_id not in wanted:
                continue
            c_terms = Counter(_tokens(chunk.text))
            length_norm = 1.0 / math.sqrt(max(sum(c_terms.values()), 1))
            score = 0.0
            for term, q_count in q_terms.items():
                tf = c_terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log((n_docs + 1) / (self._doc_freq.get(term, 0) + 1)) + 1.0
                score += q_count * tf * idf * length_norm
            if score:
                scores[chunk.chunk_id] = score
        return scores

    def _is_required(self, hit: Hit) -> bool:
        name = hit.chunk.filename.lower()
        rel = hit.chunk.relpath.lower()
        return "optional" not in name and "optional" not in rel

    def _load_reranker(self):
        if not self.cfg.reranker_model:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg.reranker_model)
        except Exception as exc:
            print(f"  ! reranker unavailable ({exc}); using hybrid scores only")
            self._reranker = False
        return self._reranker or None

    def retrieve(self, query: str, *, top_k: int | None = None,
                 modules: Iterable[str] | None = None,
                 include_optional: bool = False) -> list[Hit]:
        modules = list(modules) if modules else self.navigator.module_filter(query)
        k = top_k or self.cfg.top_k
        vector_hits = self.index.query(
            query,
            k=self.cfg.vector_k,
            min_score=RetrievalConfig().min_score,
            modules=modules,
        )
        keyword_scores = self._keyword_scores(query, modules)
        by_id: dict[str, Hit] = {h.chunk.chunk_id: h for h in vector_hits}
        for chunk in self.index.chunks:
            if chunk.chunk_id in keyword_scores and chunk.chunk_id not in by_id:
                by_id[chunk.chunk_id] = Hit(chunk=chunk, score=0.0)

        max_kw = max(keyword_scores.values(), default=1.0)
        fused: list[Hit] = []
        for hit in by_id.values():
            if not include_optional and self.cfg.prefer_required and not self._is_required(hit):
                continue
            kw = keyword_scores.get(hit.chunk.chunk_id, 0.0) / max_kw
            required_bonus = 1.0 if self._is_required(hit) else 0.0
            score = (
                self.cfg.vector_weight * hit.score
                + self.cfg.keyword_weight * kw
                + self.cfg.required_weight * required_bonus
            )
            fused.append(Hit(chunk=hit.chunk, score=float(score)))
        fused.sort(key=lambda h: h.score, reverse=True)
        fused = fused[: max(k * 3, k)]

        reranker = self._load_reranker()
        if reranker and fused:
            pairs = [(query, h.chunk.text[:1600]) for h in fused]
            rerank_scores = reranker.predict(pairs)
            fused = [Hit(chunk=h.chunk, score=float(score)) for h, score in zip(fused, rerank_scores)]
            fused.sort(key=lambda h: h.score, reverse=True)
        self._inject_missing_modules(fused, query, modules)
        return self._diversify_by_module(fused, modules, k)

    def _inject_missing_modules(self, hits: list[Hit], query: str,
                                modules: Iterable[str] | None) -> None:
        if not modules:
            return
        present = {h.chunk.module_id for h in hits}
        used = {h.chunk.chunk_id for h in hits}
        for module_id in modules:
            if module_id in present:
                continue
            scores = self._keyword_scores(query, [module_id])
            best_chunk = None
            best_score = -1.0
            for chunk in self.index.chunks:
                if chunk.module_id != module_id or chunk.chunk_id in used:
                    continue
                score = scores.get(chunk.chunk_id, 0.0)
                if score > best_score:
                    best_chunk = chunk
                    best_score = score
            if best_chunk is not None:
                hits.append(Hit(chunk=best_chunk, score=float(best_score)))
                used.add(best_chunk.chunk_id)
        hits.sort(key=lambda h: h.score, reverse=True)

    def _diversify_by_module(self, hits: list[Hit], modules: Iterable[str] | None,
                             k: int) -> list[Hit]:
        """Ensure explicit module matches are represented before filling by score."""
        if not modules:
            return hits[:k]
        selected: list[Hit] = []
        used: set[str] = set()
        for module_id in modules:
            best = next(
                (h for h in hits if h.chunk.module_id == module_id and h.chunk.chunk_id not in used),
                None,
            )
            if best is not None:
                selected.append(best)
                used.add(best.chunk.chunk_id)
            if len(selected) >= k:
                return selected
        for hit in hits:
            if hit.chunk.chunk_id in used:
                continue
            selected.append(hit)
            used.add(hit.chunk.chunk_id)
            if len(selected) >= k:
                break
        return selected

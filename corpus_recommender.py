"""Course-corpus recommender for Assured Autonomy readings.

This is a lightweight planning layer over the existing RAG index. It answers a
different question than the agent: "Which parts of the corpus should I study or
cite for this project?" rather than "What is the final answer?"
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import AA_MODULES
from .hybrid_retriever import HybridPaperRetriever, HybridRetrievalConfig
from .indexing import Hit, RagIndex
from .syllabus import SyllabusNavigator


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def _is_required(relpath: str) -> bool:
    text = relpath.lower()
    return "optional" not in text and "supplement" not in text


def _short_path(relpath: str) -> str:
    return relpath.replace("/", "\\")


@dataclass
class PassageRecommendation:
    citation: str
    page: int
    score: float
    text: str


@dataclass
class PaperRecommendation:
    rank: int
    filename: str
    relpath: str
    module_id: str
    module_name: str
    required: bool
    relevance_score: float
    diversity_adjusted_score: float
    rationale: str
    passages: list[PassageRecommendation]


@dataclass
class CorpusRecommendationReport:
    query: str
    routed_modules: list[dict]
    recommendations: list[PaperRecommendation]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "routed_modules": self.routed_modules,
            "recommendations": [
                {
                    **{k: v for k, v in asdict(rec).items() if k != "passages"},
                    "passages": [asdict(p) for p in rec.passages],
                }
                for rec in self.recommendations
            ],
        }


def recommend_corpus(
    query: str,
    *,
    index_dir: str | Path = ".aate_index",
    papers_dir: str | Path = 'papers',
    top_papers: int = 10,
    passages_per_paper: int = 2,
    candidate_chunks: int = 80,
    include_optional: bool = False,
    route_top_k: int = 6,
    diversity_weight: float = 0.18,
    required_weight: float = 0.08,
    reranker_model: str | None = "BAAI/bge-reranker-base",
) -> CorpusRecommendationReport:
    """Recommend a balanced set of papers/passages for a project question."""
    index = RagIndex.load(index_dir)
    navigator = SyllabusNavigator()
    module_matches = navigator.match(query, top_k=route_top_k)
    modules = [m.module_id for m in module_matches] or None
    routed_modules = [
        {
            "module_id": m.module_id,
            "module_name": m.module_name,
            "score": m.score,
            "evidence": m.evidence,
        }
        for m in module_matches
    ]

    retriever = HybridPaperRetriever(
        index,
        corpus_dir=papers_dir,
        cfg=HybridRetrievalConfig(
            top_k=candidate_chunks,
            vector_k=max(candidate_chunks * 2, 40),
            keyword_k=max(candidate_chunks * 2, 40),
            reranker_model=reranker_model,
            prefer_required=not include_optional,
        ),
    )
    hits = retriever.retrieve(
        query,
        top_k=candidate_chunks,
        modules=modules,
        include_optional=include_optional,
    )
    if not hits and modules:
        hits = retriever.retrieve(
            query,
            top_k=candidate_chunks,
            modules=None,
            include_optional=include_optional,
        )

    by_paper: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        by_paper[hit.chunk.relpath].append(hit)

    q_terms = Counter(_tokens(query))
    routed = {m.module_id for m in module_matches}
    candidates: list[tuple[float, str, list[Hit], str]] = []
    for relpath, paper_hits in by_paper.items():
        paper_hits.sort(key=lambda h: h.score, reverse=True)
        best = max(h.score for h in paper_hits)
        mean_top = sum(h.score for h in paper_hits[:3]) / min(3, len(paper_hits))
        coverage = min(len(paper_hits), 5) / 5.0
        required_bonus = required_weight if _is_required(relpath) else 0.0
        module_bonus = 0.06 if paper_hits[0].chunk.module_id in routed else 0.0
        score = 0.70 * best + 0.20 * mean_top + 0.04 * coverage + required_bonus + module_bonus
        rationale = _rationale(query, q_terms, paper_hits)
        candidates.append((float(score), relpath, paper_hits, rationale))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: list[tuple[float, float, str, list[Hit], str]] = []
    module_counts: Counter[str] = Counter()
    used: set[str] = set()
    while candidates and len(selected) < top_papers:
        best_idx = 0
        best_adjusted = -math.inf
        for i, (score, relpath, paper_hits, rationale) in enumerate(candidates):
            if relpath in used:
                continue
            module_id = paper_hits[0].chunk.module_id
            diversity_penalty = diversity_weight * module_counts[module_id]
            adjusted = score - diversity_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i
        score, relpath, paper_hits, rationale = candidates.pop(best_idx)
        used.add(relpath)
        module_counts[paper_hits[0].chunk.module_id] += 1
        selected.append((score, best_adjusted, relpath, paper_hits, rationale))

    recs: list[PaperRecommendation] = []
    for rank, (score, adjusted, relpath, paper_hits, rationale) in enumerate(selected, start=1):
        first = paper_hits[0].chunk
        passages = [
            PassageRecommendation(
                citation=hit.chunk.citation(),
                page=hit.chunk.page,
                score=round(hit.score, 4),
                text=_trim(hit.chunk.text, 650),
            )
            for hit in paper_hits[:passages_per_paper]
        ]
        recs.append(
            PaperRecommendation(
                rank=rank,
                filename=first.filename,
                relpath=_short_path(relpath),
                module_id=first.module_id,
                module_name=first.module_name,
                required=_is_required(relpath),
                relevance_score=round(score, 4),
                diversity_adjusted_score=round(adjusted, 4),
                rationale=rationale,
                passages=passages,
            )
        )

    return CorpusRecommendationReport(query=query, routed_modules=routed_modules, recommendations=recs)


def render_markdown(report: CorpusRecommendationReport) -> str:
    lines: list[str] = []
    lines.append("# Corpus Recommendations")
    lines.append("")
    lines.append(f"**Query:** {report.query}")
    lines.append("")
    lines.append("## Routed Modules")
    if report.routed_modules:
        for m in report.routed_modules:
            ev = ", ".join(m["evidence"])
            lines.append(f"- {m['module_id']} {m['module_name']} (score={m['score']}; evidence: {ev})")
    else:
        lines.append("- No deterministic module match; recommendations use whole-corpus retrieval.")
    lines.append("")
    lines.append("## Recommended Corpus Parts")
    for rec in report.recommendations:
        req = "Required" if rec.required else "Optional/Supplementary"
        lines.append("")
        lines.append(f"### {rec.rank}. {rec.filename}")
        lines.append(f"- Module: {rec.module_id} {rec.module_name}")
        lines.append(f"- Reading status: {req}")
        lines.append(f"- Local path: {rec.relpath}")
        lines.append(f"- Scores: relevance={rec.relevance_score}, diversified={rec.diversity_adjusted_score}")
        lines.append(f"- Why this is recommended: {rec.rationale}")
        for i, passage in enumerate(rec.passages, start=1):
            lines.append(f"- Passage {i}: {passage.citation} score={passage.score}")
            lines.append(f"  > {_one_line(passage.text)}")
    lines.append("")
    return "\n".join(lines)


def _rationale(query: str, q_terms: Counter[str], hits: list[Hit]) -> str:
    text = " ".join(h.chunk.text[:1200] for h in hits[:3]).lower()
    matches = [term for term in q_terms if term in text]
    matches = [m for m in matches if len(m) > 3]
    if matches:
        key = ", ".join(matches[:8])
        return f"High-scoring passages overlap with query concepts: {key}."
    module_name = hits[0].chunk.module_name
    return f"Retrieved as a strong {module_name} source for this project question."


def _trim(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def save_report(report: CorpusRecommendationReport, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "corpus_recommendations"
    md_path = out / f"{stem}.md"
    json_path = out / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return md_path

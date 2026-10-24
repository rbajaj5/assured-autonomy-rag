"""Compare baseline vs RAG vs RAG+rubric on the same input."""
from __future__ import annotations
from dataclasses import dataclass

from .analyze import (
    AnalysisResult, analyze_baseline, analyze_with_rag, analyze_with_rubric,
)
from .indexing import RagIndex
from .llm import LLMClient
from .loaders import NewItem
from .rubric import RUBRIC_CRITERIA, rubric_total


@dataclass
class ComparisonReport:
    item: NewItem
    baseline: AnalysisResult
    rag: AnalysisResult
    rag_rubric: AnalysisResult
    judge_summary: str        # LLM-judge meta-comparison

    def to_dict(self) -> dict:
        return {
            "item": {
                "source_kind": self.item.source_kind,
                "source": self.item.source,
                "title": self.item.title,
            },
            "baseline": self.baseline.to_dict(),
            "rag": self.rag.to_dict(),
            "rag_rubric": self.rag_rubric.to_dict(),
            "judge_summary": self.judge_summary,
        }


# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = """\
You are an evaluator comparing three analyses of the same cybersecurity item:

  A. BASELINE  — LLM with no corpus access
  B. RAG       — LLM grounded in retrieved JHU Assured Autonomy passages
  C. RAG+RUBRIC— RAG plus a structured self-rubric

Your job is to compare A, B, C honestly. Be specific. Avoid generic praise.
You are not allowed to add new domain claims; you only evaluate what is in
the three analyses. Quote at most 15 words from any single source.

Produce a markdown report with these sections:

### 1. What baseline missed
What did A fail to surface that B or C covered, and why does it matter for a
working cybersecurity consultant?

### 2. What RAG actually improved
Where did the retrieved passages add real signal to B/C — concrete concepts,
sharper threat-model framing, better recommendations? Cite specific items
from the retrieved passage list (by [Module / Filename, p. N]).

### 3. Did RAG introduce irrelevant analogies or hallucinated citations?
Flag any forced course-concept analogies, citations not present in the
retrieved set, or stretched claims. If none, say so.

### 4. Did the rubric improve usefulness?
Compare B and C. Did the explicit rubric in C produce a more useful, more
honest, or more structured analysis? Or just more boilerplate? Be specific.

### 5. Which would be more useful to a cybersecurity consultant on Monday morning?
Pick A, B, or C and justify in 3-5 sentences. If they're tied, say so.

### 6. Risks and caveats
Things the consultant should NOT take from any of these three analyses without
verification."""


def _judge_user_prompt(item: NewItem,
                        a: AnalysisResult, b: AnalysisResult, c: AnalysisResult) -> str:
    retrieved_block = "\n".join(
        f"  [R{i+1}] {h.chunk.citation()} (score={h.score:.3f})"
        for i, h in enumerate(b.retrieved)
    ) or "  (none)"
    return (
        f"ITEM\n{item.header()}\n\n"
        f"RETRIEVED PASSAGE CITATIONS USED FOR B/C (the only valid corpus citations):\n"
        f"{retrieved_block}\n\n"
        f"=== A. BASELINE OUTPUT ===\n{a.raw_output}\n\n"
        f"=== B. RAG OUTPUT ===\n{b.raw_output}\n\n"
        f"=== C. RAG+RUBRIC OUTPUT ===\n{c.raw_output}\n"
    )


# ---------------------------------------------------------------------------
def compare_modes(item: NewItem, index: RagIndex, llm: LLMClient, *,
                  judge_llm: LLMClient | None = None) -> ComparisonReport:
    """Run all three modes and produce a comparison.

    If `judge_llm` is None, uses `llm` as the judge (slight self-evaluation
    bias; for rigorous evaluation pass a different model as `judge_llm`).
    """
    a = analyze_baseline(item, llm)
    b = analyze_with_rag(item, index, llm)
    c = analyze_with_rubric(item, index, llm)

    judge = judge_llm or llm
    summary = judge.complete(_JUDGE_SYSTEM, _judge_user_prompt(item, a, b, c))
    return ComparisonReport(item=item, baseline=a, rag=b, rag_rubric=c,
                            judge_summary=summary)


# ---------------------------------------------------------------------------
def render_comparison_markdown(report: ComparisonReport) -> str:
    """Build a single markdown report combining all four pieces."""
    title = report.item.title or report.item.source
    parts: list[str] = []
    parts.append(f"# AATE Comparison Report\n\n**Item:** {title}\n")
    parts.append(f"**Source:** {report.item.source_kind} — `{report.item.source}`\n")
    parts.append(f"**LLM:** {report.baseline.llm_provider}\n")
    parts.append(f"**Item length:** {len(report.item.text):,} chars\n")

    # Retrieval audit
    parts.append("\n## Retrieved passages (used by RAG and RAG+RUBRIC)\n")
    if report.rag.retrieved:
        parts.append("| Rank | Module | File | Page | Score |")
        parts.append("|---|---|---|---|---|")
        for i, h in enumerate(report.rag.retrieved, start=1):
            parts.append(
                f"| R{i} | {h.chunk.module_id} | `{h.chunk.filename}` "
                f"| {h.chunk.page} | {h.score:.3f} |"
            )
    else:
        parts.append("_(no passages retrieved)_")

    # Rubric (rag_rubric only)
    parts.append("\n## Rubric self-scores (RAG+RUBRIC mode)\n")
    if report.rag_rubric.rubric_scores:
        scores = report.rag_rubric.rubric_scores
        parts.append("| Criterion | Score (1-5) |")
        parts.append("|---|---|")
        for c in RUBRIC_CRITERIA:
            parts.append(f"| {c.label} | {scores.get(c.key, '—')} |")
        total = rubric_total(scores)
        if total is not None:
            parts.append(
                f"| **Total** | **{total} / {5 * len(RUBRIC_CRITERIA)}** |"
            )
    else:
        parts.append("_(no rubric scores parsed; check the raw output)_")

    # Judge summary
    parts.append("\n## Mode comparison (LLM judge)\n")
    parts.append(report.judge_summary or "_(judge produced no output)_")

    # Raw outputs (collapsed sections)
    parts.append("\n---\n\n## Raw outputs\n")
    for label, res in [
        ("A. Baseline", report.baseline),
        ("B. RAG", report.rag),
        ("C. RAG + rubric", report.rag_rubric),
    ]:
        parts.append(f"\n### {label}\n")
        parts.append("```\n" + res.raw_output.strip() + "\n```\n")

    return "\n".join(parts)

"""The three analysis modes:

  - analyze_baseline      : LLM-only, no corpus access. (Control.)
  - analyze_with_rag      : Retrieve from corpus, ground answer, cite passages.
  - analyze_with_rubric   : RAG + structured rubric self-scoring.

The output structure is identical across modes so they are directly comparable.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

from .indexing import RagIndex, Hit, format_hits_for_prompt
from .config import RetrievalConfig
from .llm import LLMClient
from .rubric import RUBRIC_CRITERIA, parse_rubric_scores
from .loaders import NewItem


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    mode: str                                  # baseline | rag | rag_rubric
    item: NewItem
    raw_output: str                            # full LLM response
    retrieved: list[Hit] = field(default_factory=list)   # empty for baseline
    rubric_scores: Optional[dict] = None       # only for rag_rubric
    llm_provider: str = ""

    def to_dict(self) -> dict:
        d = {
            "mode": self.mode,
            "item": {
                "source_kind": self.item.source_kind,
                "source": self.item.source,
                "title": self.item.title,
                "text_len": len(self.item.text),
            },
            "raw_output": self.raw_output,
            "rubric_scores": self.rubric_scores,
            "llm_provider": self.llm_provider,
            "retrieved": [
                {
                    "citation": h.chunk.citation(),
                    "module_id": h.chunk.module_id,
                    "module_name": h.chunk.module_name,
                    "filename": h.chunk.filename,
                    "page": h.chunk.page,
                    "score": h.score,
                    "chunk_id": h.chunk.chunk_id,
                }
                for h in self.retrieved
            ],
        }
        return d


# ---------------------------------------------------------------------------
# System / user prompts
# ---------------------------------------------------------------------------
_OUTPUT_SECTIONS = """\
Produce exactly these 10 sections, in order, using the section headers verbatim:

1. Executive summary
2. Core cybersecurity or autonomy-safety risk
3. Relevant Assured Autonomy module(s)
4. Relevant readings from the corpus, with citations
5. Threat model
   - Attacker or failure mode
   - Assets at risk
   - Assumptions
   - Adversarial vs non-adversarial conditions
6. Transfer analysis
   - Course concepts that transfer cleanly
   - Course concepts that do not transfer / weak analogies
7. Practical recommendations for a cybersecurity consultant
8. Questions to ask the client
9. Monitoring / testing / governance implications
10. Uncertainty and overclaim warnings"""


_AA_MODULE_LIST = """\
The JHU Assured Autonomy modules are:
M01-02 Introduction; M03 Safety, Security, Privacy; M04 Anomaly and Fault Detection;
M05 Dataset Shift; M06 Verification and Validation; M07 Test and Evaluation;
M08 Interpretable Autonomy; M09 Human-AI Integration; M10 Policy and Governance;
M11 Adversarial Autonomy; M12 Runtime Assurance; M13 Safe Reinforcement Learning;
M14 Multi-Agent and Cooperative Systems."""


_BASELINE_SYSTEM = f"""\
You are a careful cybersecurity analyst advising a working consultant.

You DO NOT have access to the JHU Assured Autonomy course corpus in this run.
This is the BASELINE condition.

Rules:
- For section 4 ("Relevant readings"), say: "No corpus consulted in baseline mode."
- For section 3, you may name likely-relevant module IDs from the official list,
  but make clear this is from your general knowledge of such curricula.
- Be defensive-only. Do not give offensive cyber instructions.
- Keep direct quotes from the input under 15 words.
- Separate facts that come from the input item from inferences you are making.
- If you are uncertain, say so explicitly in section 10.

{_AA_MODULE_LIST}

{_OUTPUT_SECTIONS}"""


_RAG_SYSTEM = f"""\
You are a careful cybersecurity analyst advising a working consultant. You are
grounded in retrieved passages from the JHU Assured Autonomy course corpus.

Rules:
- Cite ONLY the retrieved passages provided to you. Use citations of the form
  [Module / Filename, p. N], exactly as they appear in the retrieved-passage
  block. Do not invent citations or page numbers.
- If a relevant passage is not present, say: "Corpus does not directly support
  this in the retrieved set."
- Direct quotes from any source must be under 15 words. Default to paraphrase.
- Clearly separate (a) facts from the new item, (b) concepts grounded in the
  retrieved corpus, and (c) your own inferences/analogies.
- Be defensive-only. Do not give offensive cyber instructions.
- Avoid forced analogies: if a course concept does not cleanly map, say so in
  section 6 rather than stretching it.

{_AA_MODULE_LIST}

{_OUTPUT_SECTIONS}"""


_RUBRIC_TAIL = f"""\

After the 10 sections, on a new line, output:

RUBRIC SCORES (JSON)

followed by a single JSON object on the next line scoring YOUR OWN analysis
above on each criterion from 1 (poor) to 5 (excellent). Use exactly these keys:

{ {c.key: 0 for c in RUBRIC_CRITERIA} }

Criteria definitions:
""" + "\n".join(f"- {c.key}: {c.description}" for c in RUBRIC_CRITERIA) + """

The rubric scores are an honest self-assessment, not a pep talk. If your
analysis is weak in some dimension, score it low and say why in section 10."""


# ---------------------------------------------------------------------------
# Mode 1: Baseline
# ---------------------------------------------------------------------------
def analyze_baseline(item: NewItem, llm: LLMClient, *,
                     max_input_chars: int = 18000) -> AnalysisResult:
    """Baseline: LLM-only, no corpus."""
    body = item.text[:max_input_chars]
    user = (
        f"NEW ITEM METADATA:\n{item.header()}\n\n"
        f"NEW ITEM TEXT (truncated to {len(body)} chars of "
        f"{len(item.text)} total):\n---\n{body}\n---\n"
    )
    out = llm.complete(_BASELINE_SYSTEM, user)
    return AnalysisResult(
        mode="baseline", item=item, raw_output=out, llm_provider=llm.name
    )


# ---------------------------------------------------------------------------
# Mode 2: RAG-grounded
# ---------------------------------------------------------------------------
def analyze_with_rag(item: NewItem, index: RagIndex, llm: LLMClient, *,
                     retrieval: RetrievalConfig | None = None,
                     max_input_chars: int = 14000,
                     query_chars: int = 3000) -> AnalysisResult:
    """RAG: retrieve relevant passages, ground answer, cite."""
    rcfg = retrieval or RetrievalConfig()
    # Use the head of the item (often the abstract / lede) as the retrieval query.
    query = item.text[:query_chars]
    hits = index.query(query, k=rcfg.top_k, min_score=rcfg.min_score)
    passages = format_hits_for_prompt(hits)
    body = item.text[:max_input_chars]

    user = (
        f"NEW ITEM METADATA:\n{item.header()}\n\n"
        f"RETRIEVED PASSAGES FROM ASSURED AUTONOMY CORPUS:\n{passages}\n\n"
        f"NEW ITEM TEXT (truncated to {len(body)} chars of "
        f"{len(item.text)} total):\n---\n{body}\n---\n"
    )
    out = llm.complete(_RAG_SYSTEM, user)
    return AnalysisResult(
        mode="rag", item=item, raw_output=out, retrieved=hits,
        llm_provider=llm.name,
    )


# ---------------------------------------------------------------------------
# Mode 3: RAG + structured rubric
# ---------------------------------------------------------------------------
def analyze_with_rubric(item: NewItem, index: RagIndex, llm: LLMClient, *,
                        retrieval: RetrievalConfig | None = None,
                        max_input_chars: int = 14000,
                        query_chars: int = 3000) -> AnalysisResult:
    """RAG + rubric: same as RAG, but the LLM also self-scores against a fixed rubric."""
    rcfg = retrieval or RetrievalConfig()
    query = item.text[:query_chars]
    hits = index.query(query, k=rcfg.top_k, min_score=rcfg.min_score)
    passages = format_hits_for_prompt(hits)
    body = item.text[:max_input_chars]

    system = _RAG_SYSTEM + _RUBRIC_TAIL
    user = (
        f"NEW ITEM METADATA:\n{item.header()}\n\n"
        f"RETRIEVED PASSAGES FROM ASSURED AUTONOMY CORPUS:\n{passages}\n\n"
        f"NEW ITEM TEXT (truncated to {len(body)} chars of "
        f"{len(item.text)} total):\n---\n{body}\n---\n"
    )
    out = llm.complete(system, user)
    scores = parse_rubric_scores(out)
    return AnalysisResult(
        mode="rag_rubric", item=item, raw_output=out, retrieved=hits,
        rubric_scores=scores, llm_provider=llm.name,
    )

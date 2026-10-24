"""Iterative RAG: decompose, retrieve, synthesize, critique, revise."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .agentic_rag import (
    _citation_line,
    _expand_citation_handles,
    _trim_redundant_full_citations,
)
from .hybrid_retriever import HybridPaperRetriever, HybridRetrievalConfig
from .indexing import Hit, RagIndex
from .llm import LLMClient, get_llm
from .paths import DEFAULT_INDEX_DIR, PAPERS_DIR
from .prompt_router import route_prompt
from .syllabus import SyllabusNavigator


@dataclass
class Probe:
    probe_id: str
    question: str
    purpose: str


@dataclass
class IterativeRagConfig:
    index_dir: Path = DEFAULT_INDEX_DIR
    papers_dir: Path = PAPERS_DIR
    provider: str = "openai"
    model: str | None = "gpt-4.1"
    top_k_per_probe: int = 4
    max_probes: int = 6
    synthesis_max_tokens: int = 6000
    critique_max_tokens: int = 1800
    revise: bool = True
    out_dir: Path = Path("iterative_runs")


class IterativeRagRunner:
    def __init__(self, cfg: IterativeRagConfig, llm: LLMClient | None = None):
        self.cfg = cfg
        self.llm = llm or get_llm(cfg.provider, model=cfg.model)
        self.index = RagIndex.load(cfg.index_dir)
        self.navigator = SyllabusNavigator()
        self.retriever = HybridPaperRetriever(
            self.index,
            corpus_dir=cfg.papers_dir,
            cfg=HybridRetrievalConfig(top_k=cfg.top_k_per_probe),
        )

    def run(self, query: str) -> dict:
        routed = route_prompt(query)
        probes = self._decompose(query, routed.routed_query)
        retrieval = self._retrieve(probes)
        answer = self._synthesize(query, routed.routed_query, probes, retrieval)
        critique = self._critique(query, answer)
        if self.cfg.revise and not critique.get("sufficient", False):
            answer = self._revise(query, answer, critique, probes, retrieval)
            critique = self._critique(query, answer)

        result = {
            "query": query,
            "routed_prompt": routed.to_dict(),
            "probes": [asdict(p) for p in probes],
            "retrieval": self._retrieval_to_dict(retrieval),
            "answer": answer,
            "critique": critique,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }
        self._save(result)
        return result

    def _decompose(self, query: str, routed_query: str) -> list[Probe]:
        system = (
            "You decompose user questions into RAG probes for an Assured Autonomy "
            "course corpus. Return only JSON with key `probes`. Each probe needs "
            "`question` and `purpose`. Make probes concrete, non-overlapping, and "
            "retrieval-friendly. Include a probe for what not to infer when an "
            "unknown proper noun or policy-specific system appears."
        )
        user = (
            f"Original question:\n{query}\n\n"
            f"Routed/expanded question:\n{routed_query}\n\n"
            f"Return at most {self.cfg.max_probes} probes."
        )
        raw = self.llm.complete(system, user, temperature=0.1, max_tokens=1200)
        data = _json_from_text(raw)
        probes: list[Probe] = []
        for i, item in enumerate(data.get("probes", []), start=1):
            if len(probes) >= self.cfg.max_probes:
                break
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            probes.append(Probe(
                probe_id=f"P{i}",
                question=question,
                purpose=str(item.get("purpose", "")).strip() or "Retrieve relevant course evidence.",
            ))
        if probes:
            return probes
        return [
            Probe("P1", routed_query, "Retrieve broad course evidence for the routed question."),
            Probe("P2", "What should not be inferred without system-specific policy documents?", "Prevent overclaiming."),
        ]

    def _retrieve(self, probes: list[Probe]) -> dict[str, list[Hit]]:
        retrieval: dict[str, list[Hit]] = {}
        for probe in probes:
            modules = self.navigator.module_filter(probe.question, top_k=4)
            retrieval[probe.probe_id] = self.retriever.retrieve(
                probe.question,
                top_k=self.cfg.top_k_per_probe,
                modules=modules,
            )
        return retrieval

    def _synthesize(self, query: str, routed_query: str, probes: list[Probe],
                    retrieval: dict[str, list[Hit]]) -> str:
        cite_map, evidence = self._evidence_block(retrieval)
        probe_block = "\n".join(
            f"{p.probe_id}. {p.question} Purpose: {p.purpose}" for p in probes
        )
        system = (
            "You synthesize an iterative RAG answer. Use the probe answers/evidence "
            "to answer the original question. Use citation handles like [CITE_1], "
            "not raw paths. Do not invent policy for unknown systems. If a system is "
            "hypothetical or undefined, say so. Be concise but complete."
        )
        user = (
            f"Original question:\n{query}\n\n"
            f"Routed query:\n{routed_query}\n\n"
            f"Probes:\n{probe_block}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Final answer requirements:\n"
            "- First answer the original question.\n"
            "- Separate denial reasons from evidence the requester could provide.\n"
            "- Cite every major reason with a [CITE_N] handle.\n"
            "- Include a short caveat about not knowing actual Jureko policy unless provided.\n"
        )
        raw = self._complete_nonempty(
            system,
            user,
            temperature=0.1,
            max_tokens=self.cfg.synthesis_max_tokens,
            stage="synthesis",
        )
        expanded = _expand_citation_handles(raw, cite_map)
        return _trim_redundant_full_citations(expanded)

    def _critique(self, query: str, answer: str) -> dict:
        system = (
            "You are a strict evaluator for an Assured Autonomy RAG answer. Return "
            "only JSON with keys: sufficient (boolean), missing_points (array), "
            "overclaims (array), citation_issues (array), revision_instruction (string)."
        )
        user = (
            f"Original question:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            "Check if the answer addresses the question, avoids unsupported policy claims, "
            "and gives citation-backed reasons and mitigation/evidence items."
        )
        raw = self.llm.complete(
            system,
            user,
            temperature=0.0,
            max_tokens=self.cfg.critique_max_tokens,
        )
        return _json_from_text(raw)

    def _revise(self, query: str, answer: str, critique: dict, probes: list[Probe],
                retrieval: dict[str, list[Hit]]) -> str:
        cite_map, evidence = self._evidence_block(retrieval)
        system = (
            "Revise the RAG answer using the critique. Use only [CITE_N] handles "
            "from the evidence. Do not write raw paths. Do not invent actual policy."
        )
        user = (
            f"Original question:\n{query}\n\n"
            f"Draft answer:\n{answer}\n\n"
            f"Critique:\n{json.dumps(critique, indent=2)}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Produce the revised final answer."
        )
        raw = self._complete_nonempty(
            system,
            user,
            temperature=0.1,
            max_tokens=self.cfg.synthesis_max_tokens,
            stage="revision",
        )
        expanded = _expand_citation_handles(raw, cite_map)
        return _trim_redundant_full_citations(expanded)

    def _complete_nonempty(self, system: str, user: str, *, temperature: float,
                           max_tokens: int, stage: str) -> str:
        """Retry long synthesis calls before allowing a blank artifact through."""
        budgets = [max_tokens, max(max_tokens * 2, 10000)]
        for budget in budgets:
            raw = self.llm.complete(
                system,
                user,
                temperature=temperature,
                max_tokens=budget,
            ).strip()
            if raw:
                return raw
        raise RuntimeError(
            f"LLM returned an empty {stage} after retrying. Try a larger "
            "--synthesis-max-tokens value, a smaller --max-probes value, or a "
            "non-reasoning model such as gpt-4.1 for this step."
        )

    def _evidence_block(self, retrieval: dict[str, list[Hit]]) -> tuple[dict[str, Hit], str]:
        cite_map: dict[str, Hit] = {}
        lines: list[str] = []
        seen_chunks: set[str] = set()
        n = 0
        for probe_id, hits in retrieval.items():
            lines.append(f"\n## Evidence for {probe_id}")
            for hit in hits:
                if hit.chunk.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(hit.chunk.chunk_id)
                n += 1
                handle = f"CITE_{n}"
                cite_map[handle] = hit
                lines.append(
                    f"[{handle}] Source: {_source(hit)}; Module: {hit.chunk.module_name}; "
                    f"Local Path: {self.cfg.papers_dir / hit.chunk.relpath}\n"
                    f"{hit.chunk.text[:900]}"
                )
        return cite_map, "\n\n".join(lines)

    def _retrieval_to_dict(self, retrieval: dict[str, list[Hit]]) -> dict:
        return {
            probe_id: [
                {
                    "module": h.chunk.module_name,
                    "filename": h.chunk.filename,
                    "path": str(self.cfg.papers_dir / h.chunk.relpath),
                    "score": h.score,
                }
                for h in hits
            ]
            for probe_id, hits in retrieval.items()
        }

    def _save(self, result: dict) -> None:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = self.cfg.out_dir / f"iterative_rag_{_safe_name(self.cfg.provider)}_{_safe_name(self.cfg.model or 'default')}_{stamp}"
        json_path = Path(f"{stem}.json")
        md_path = Path(f"{stem}.md")
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")


def _source(hit: Hit) -> str:
    parts = Path(hit.chunk.filename).stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return Path(hit.chunk.filename).stem


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return safe.strip("-") or "default"


def _json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _render_markdown(result: dict) -> str:
    lines = [
        "# Iterative RAG Run",
        "",
        "## Query",
        result["query"],
        "",
        "## Probes",
    ]
    for probe in result["probes"]:
        lines.append(f"- **{probe['probe_id']}** {probe['question']} ({probe['purpose']})")
    lines.extend(["", "## Answer", result["answer"], "", "## Critique", "```json"])
    lines.append(json.dumps(result["critique"], indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"

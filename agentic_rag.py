"""ReAct-style agent for module-aware Assured Autonomy RAG."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .hybrid_retriever import HybridPaperRetriever, HybridRetrievalConfig
from .indexing import RagIndex
from .llm import LLMClient, get_llm
from .paths import DEFAULT_INDEX_DIR, PAPERS_DIR
from .prompt_router import route_prompt
from .syllabus import SyllabusNavigator


def _author_year(filename: str) -> str:
    parts = Path(filename).stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return Path(filename).stem


def _citation_line(hit) -> str:
    return (
        f"Evidence from {hit.chunk.module_name}. "
        f"| Source: {_author_year(hit.chunk.filename)} | Module: {hit.chunk.module_name} | "
        f"Local Path: {PAPERS_DIR / hit.chunk.relpath}"
    )


def _citation_suffix(hit) -> str:
    return (
        f"| Source: {_author_year(hit.chunk.filename)} | Module: {hit.chunk.module_name} | "
        f"Local Path: {PAPERS_DIR / hit.chunk.relpath}"
    )


def _cited_modules(answer: str) -> set[str]:
    modules: set[str] = set()
    pattern = r"Source:\s*[^|\n]+\|\s*Module:\s*([^|\n\]]+)\|\s*Local Path:\s*.*?\.pdf"
    for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
        modules.add(match.group(1).strip())
    return modules


def _expand_citation_handles(answer: str, cite_map: dict[str, object]) -> str:
    """Replace model-friendly [CITE_N] handles with exact deterministic citations."""
    seen_handles: set[str] = set()

    def repl(match: re.Match) -> str:
        handle = match.group(1).upper()
        if handle in seen_handles:
            return ""
        hit = cite_map.get(handle)
        if hit is None:
            return match.group(0)
        seen_handles.add(handle)
        return _citation_suffix(hit)

    expanded = re.sub(r"\[(CITE_\d+)\]", repl, answer, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", expanded).strip()


def _trim_redundant_full_citations(answer: str) -> str:
    """Drop repeated full citations for the same source/module/path triple."""
    pattern = re.compile(
        r"\|\s*Source:\s*([^|\n]+)\|\s*Module:\s*([^|\n]+)\|\s*Local Path:\s*(.*?\.pdf)",
        flags=re.IGNORECASE,
    )
    seen: set[tuple[str, str, str]] = set()

    def repl(match: re.Match) -> str:
        key = tuple(part.strip().lower() for part in match.groups())
        if key in seen:
            return ""
        seen.add(key)
        return match.group(0)

    trimmed = pattern.sub(repl, answer)
    return re.sub(r"\s{2,}", " ", trimmed).strip()


@dataclass
class AgenticRagConfig:
    index_dir: Path = DEFAULT_INDEX_DIR
    papers_dir: Path = PAPERS_DIR
    provider: str = "stub"
    model: str | None = None
    top_k: int = 8
    include_optional: bool = False
    route_prompt: bool = False


class AgenticRagAgent:
    """Small ReAct agent with Paper_Retriever and Syllabus_Navigator tools."""

    def __init__(self, cfg: AgenticRagConfig | None = None,
                 llm: LLMClient | None = None):
        self.cfg = cfg or AgenticRagConfig()
        self.llm = llm or get_llm(self.cfg.provider, model=self.cfg.model)
        self.index = RagIndex.load(self.cfg.index_dir)
        self.navigator = SyllabusNavigator()
        self.retriever = HybridPaperRetriever(
            self.index,
            corpus_dir=self.cfg.papers_dir,
            cfg=HybridRetrievalConfig(top_k=self.cfg.top_k),
        )

    def answer(self, query: str) -> str:
        routed = route_prompt(query) if self.cfg.route_prompt else None
        retrieval_query = routed.routed_query if routed else query
        module_matches = self.navigator.match(retrieval_query, top_k=4)
        modules = [m.module_id for m in module_matches] or None
        hits = self.retriever.retrieve(
            retrieval_query,
            top_k=self.cfg.top_k,
            modules=modules,
            include_optional=self.cfg.include_optional,
        )
        module_block = "\n".join(
            f"- {m.module_id}: {m.module_name} (evidence: {', '.join(m.evidence)})"
            for m in module_matches
        ) or "- No confident module match; use retrieval evidence cautiously."
        cite_map = {f"CITE_{i}": h for i, h in enumerate(hits, start=1)}
        source_block = "\n\n".join(
            f"[CITE_{i}] Source: {_author_year(h.chunk.filename)}; File: {h.chunk.filename}; "
            f"Module: {h.chunk.module_name}; Local Path: {self.cfg.papers_dir / h.chunk.relpath}\n"
            f"{h.chunk.text[:1200]}"
            for i, h in enumerate(hits, start=1)
        ) or "(no retrieved passages)"

        system = (
            "You are a ReAct-style Assured Autonomy RAG agent. Use the "
            "Syllabus_Navigator observations to select module context, then use "
            "Paper_Retriever observations as evidence. Prioritize Required readings "
            "over Optional readings unless the user explicitly asks for supplementary "
            "material.\n\n"
            "Use citation handles, not raw file paths. When you use evidence from an "
            "observation, end the sentence or bullet with the matching handle such as "
            "[CITE_1]. Do not write Windows paths yourself; the software will expand "
            "citation handles after you answer.\n\n"
            "Hard output rules:\n"
            "- Do not create a separate Sources, References, or Bibliography section.\n"
            "- Do not invent source names, years, module names, or local paths.\n"
            "- Use only handles that appear in Paper_Retriever observations: [CITE_1], [CITE_2], etc.\n"
            "- Every paragraph that compares course concepts must end with at least one citation handle.\n"
            "- For comparison questions, cite at least one source from each module named "
            "by Syllabus_Navigator.\n"
            "- Keep the answer concise: 2-5 paragraphs or bullets."
        )
        user = (
            f"Question:\n{query}\n\n"
            + (f"Prompt routing analysis:\n{routed.to_dict()}\n\n" if routed else "")
            + f"Thought: map the question to course modules.\n"
            + f"Tool: Syllabus_Navigator\nObservation:\n{module_block}\n\n"
            + f"Thought: retrieve required local papers with hybrid search and reranking.\n"
            + f"Tool: Paper_Retriever\nObservation:\n{source_block}\n\n"
            + "Final Answer:"
        )
        answer = self.llm.complete(system, user, temperature=0.1)
        answer = _expand_citation_handles(answer, cite_map)
        answer = _trim_redundant_full_citations(answer)
        return self._repair_citations(answer, module_matches, hits)

    def _repair_citations(self, answer: str, module_matches, hits) -> str:
        cited = _cited_modules(answer)
        missing_lines: list[str] = []
        for match in module_matches:
            if any(match.module_name.lower() in module.lower() for module in cited):
                continue
            hit = next((h for h in hits if h.chunk.module_id == match.module_id), None)
            if hit is not None:
                missing_lines.append(_citation_line(hit))
        if not missing_lines:
            return answer
        repair_block = "\n\nRequired citation coverage:\n" + "\n".join(
            f"- {line}" for line in missing_lines
        )
        return answer.rstrip() + repair_block

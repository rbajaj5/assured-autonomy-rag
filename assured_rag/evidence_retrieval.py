from __future__ import annotations

from pathlib import Path
from typing import Any

from .documents import ingest_paths
from .retrieval import LocalRetriever, RetrievedChunk


SUPPORT_RANK = {"missing": 0, "weak": 1, "partial": 2, "strong": 3}


def locate_paper_card_roots(base: Path = Path(".")) -> list[Path]:
    roots = []
    for path in base.rglob("*"):
        if path.is_dir() and path.name in {"paper_cards", "readable_cards"}:
            roots.append(path)
    return sorted(roots)


class EvidenceRetriever:
    def __init__(
        self,
        *,
        paper_card_roots: list[Path] | None = None,
        raw_roots: list[Path] | None = None,
        paper_cards_first: bool = True,
        fallback_to_raw: bool = True,
        use_embeddings: bool = True,
    ) -> None:
        self.paper_cards_first = paper_cards_first
        self.fallback_to_raw = fallback_to_raw
        self.paper_card_roots = paper_card_roots if paper_card_roots is not None else locate_paper_card_roots()
        self.raw_roots = raw_roots if raw_roots is not None else [Path("docs"), Path("readings"), Path("corpus")]
        self.paper_chunks = ingest_paths(self.paper_card_roots) if self.paper_cards_first else []
        self.raw_chunks = ingest_paths(self.raw_roots)
        self.paper_retriever = LocalRetriever(self.paper_chunks, use_embeddings=use_embeddings)
        self.raw_retriever = LocalRetriever(self.raw_chunks, use_embeddings=use_embeddings)

    def retrieve(self, spec: dict[str, Any], *, k: int) -> dict[str, Any]:
        paper_hits = _retrieve_grounding(spec, retriever=self.paper_retriever, k=k, source_tier="paper_card") if self.paper_cards_first else []
        if self.paper_cards_first and spec.get("conceptual_aliases"):
            paper_hits = _merge_evidence(
                paper_hits,
                _retrieve_crosswalk_grounding(spec, retriever=self.paper_retriever, k=k, source_tier="paper_card"),
            )[:k]
        paper_support = support_level_from_evidence(paper_hits)
        use_raw = (not self.paper_cards_first) or (
            self.fallback_to_raw and paper_support in {"weak", "missing"}
        )
        raw_hits = _retrieve_grounding(spec, retriever=self.raw_retriever, k=k, source_tier="raw_reading") if use_raw else []
        if use_raw and spec.get("conceptual_aliases"):
            raw_hits = _merge_evidence(
                raw_hits,
                _retrieve_crosswalk_grounding(spec, retriever=self.raw_retriever, k=k, source_tier="raw_reading"),
            )[:k]
        raw_support = support_level_from_evidence(raw_hits)
        combined = _best_support(paper_support, raw_support)
        evidence = paper_hits if paper_support in {"partial", "strong"} else (paper_hits + raw_hits)
        for item in evidence:
            item["support_level"] = paper_support if item["source_tier"] == "paper_card" else raw_support
        return {
            "evidence": sorted(evidence, key=lambda item: (SUPPORT_RANK.get(item.get("support_level", "missing"), 0), item.get("score", 0.0)), reverse=True)[:k],
            "support_level": combined,
            "paper_card_support_level": paper_support,
            "raw_support_level": raw_support,
            "source_tier_counts": {
                "paper_card": len([item for item in evidence if item.get("source_tier") == "paper_card"]),
                "raw_reading": len([item for item in evidence if item.get("source_tier") == "raw_reading"]),
            },
            "paper_card_roots": [str(path) for path in self.paper_card_roots],
            "paper_card_chunk_count": len(self.paper_chunks),
            "raw_chunk_count": len(self.raw_chunks),
            "fallback_used": bool(raw_hits),
        }


def support_level_from_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "missing"
    direct = [item for item in evidence if item.get("support_relation") == "direct"]
    crosswalk = [
        item for item in evidence
        if item.get("support_relation") == "crosswalk" and item.get("relevance") in {"high", "medium"}
    ]
    indirect = [
        item for item in evidence
        if item.get("support_relation") == "indirect" and item.get("relevance") in {"high", "medium"}
    ]
    independent_crosswalk_sources = {
        item.get("filename", "") for item in crosswalk if item.get("filename")
    }
    if direct:
        return "strong"
    if len(crosswalk) >= 3 and len(independent_crosswalk_sources) >= 2:
        return "strong"
    if crosswalk or len(indirect) >= 2:
        return "partial"
    return "weak"


def _retrieve_grounding(spec: dict[str, Any], *, retriever: LocalRetriever, k: int, source_tier: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for query in spec.get("queries", []):
        for hit in retriever.search(query, k=k):
            citation = _grounding_from_hit(hit, spec, source_tier=source_tier)
            key = (citation["filename"], citation["chunk_id"])
            if key not in merged or citation["score"] > merged[key]["score"]:
                merged[key] = citation
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:k]


def _retrieve_crosswalk_grounding(spec: dict[str, Any], *, retriever: LocalRetriever, k: int, source_tier: str) -> list[dict[str, Any]]:
    crosswalk_spec = dict(spec)
    aliases = spec.get("conceptual_aliases", [])
    crosswalk_spec["queries"] = [str(alias) for alias in aliases]
    hits = _retrieve_grounding(crosswalk_spec, retriever=retriever, k=k, source_tier=source_tier)
    for hit in hits:
        hit["crosswalk_match"] = True
        hit["modern_blind_spot_label"] = spec.get("modern_blind_spot_label", "")
        hit["older_reading_list_concepts"] = spec.get("older_reading_list_concepts", [])
        hit["matched_aliases"] = _matched_aliases(hit.get("actual_excerpt") or hit.get("excerpt") or "", aliases)
        hit["mapped_older_concepts"] = _mapped_older_concepts(hit.get("actual_excerpt") or hit.get("excerpt") or "", spec)
        _apply_evidence_hygiene(hit, spec)
    return hits


def _merge_evidence(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in primary + secondary:
        key = (item.get("filename", ""), item.get("chunk_id", ""))
        if key not in merged or item.get("score", 0.0) > merged[key].get("score", 0.0):
            merged[key] = item
    return sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)


def _grounding_from_hit(hit: RetrievedChunk, spec: dict[str, Any], *, source_tier: str) -> dict[str, Any]:
    citation = hit.chunk.citation(excerpt_chars=300)
    actual_excerpt = citation.get("excerpt", "")
    citation.update({
        "actual_excerpt": actual_excerpt,
        "score": round(float(hit.score), 4),
        "method": hit.method,
        "source_tier": source_tier,
        "relevance": _citation_relevance(hit, spec),
        "concept_supported": spec.get("assurance_concept_missed", ""),
        "inferred_concept": spec.get("assurance_concept_missed", ""),
    })
    _apply_evidence_hygiene(citation, spec)
    return citation


def _citation_relevance(hit: RetrievedChunk, spec: dict[str, Any]) -> str:
    text = hit.chunk.text.lower()
    concept_terms = set()
    for query in spec.get("queries", []):
        concept_terms.update(term for term in query.lower().split() if len(term) > 4)
    overlap = sum(1 for term in concept_terms if term in text)
    if overlap >= 5:
        return "high"
    if overlap >= 2:
        return "medium"
    return "low"


def _best_support(a: str, b: str) -> str:
    return a if SUPPORT_RANK.get(a, 0) >= SUPPORT_RANK.get(b, 0) else b


def _apply_evidence_hygiene(citation: dict[str, Any], spec: dict[str, Any]) -> None:
    text = citation.get("actual_excerpt") or citation.get("excerpt") or ""
    citation["actual_excerpt"] = text
    citation["inferred_concept"] = citation.get("inferred_concept") or spec.get("assurance_concept_missed", "")
    relation = _support_relation(text, spec, bool(citation.get("crosswalk_match")))
    citation["support_relation"] = relation
    citation["explanation"] = _support_explanation(relation, citation, spec)
    if "matched_aliases" not in citation:
        citation["matched_aliases"] = _matched_aliases(text, spec.get("conceptual_aliases", []))
    if "mapped_older_concepts" not in citation:
        citation["mapped_older_concepts"] = _mapped_older_concepts(text, spec)


def _support_relation(text: str, spec: dict[str, Any], crosswalk_match: bool) -> str:
    lowered = text.lower()
    if crosswalk_match and spec.get("modern_blind_spot_label"):
        return "crosswalk"
    if _has_direct_support(lowered, spec):
        return "direct"
    if crosswalk_match or _matched_aliases(text, spec.get("conceptual_aliases", [])):
        return "crosswalk"
    if _has_indirect_support(lowered):
        return "indirect"
    return "weak"


def _has_direct_support(text: str, spec: dict[str, Any]) -> bool:
    concept = (spec.get("assurance_concept_missed") or "").lower()
    direct_groups = [
        ["runtime assurance", "fallback", "safe controller", "safety envelope", "switch"],
        ["safety case", "assurance case", "claim", "evidence", "argument"],
        ["human oversight", "human in the loop", "operator", "human fallback"],
        ["operational output", "operational recommendation", "execution pathway", "decision support"],
        ["non-interference", "sandbox", "containment", "live telemetry", "actuation boundary"],
    ]
    if "labels such as" in concept or "containment" in concept:
        safety_label_terms = [
            "unsupported assurance claim",
            "unsupported safety claim",
            "safety case scope",
            "operational context",
            "test environment",
            "deployment environment",
            "evidence does not support",
            "non-interference",
            "sandbox boundary",
            "containment evidence",
            "overclaim",
        ]
        return any(term in text for term in safety_label_terms)
    return any(sum(1 for term in group if term in text) >= 2 for group in direct_groups)


def _has_indirect_support(text: str) -> bool:
    terms = [
        "assurance",
        "safety",
        "risk",
        "hazard",
        "monitor",
        "validation",
        "verification",
        "deployment",
        "operational",
        "controller",
    ]
    return sum(1 for term in terms if term in text) >= 2


def _matched_aliases(text: str, aliases: list[str]) -> list[str]:
    lowered = text.lower()
    return [alias for alias in aliases if alias.lower() in lowered]


def _mapped_older_concepts(text: str, spec: dict[str, Any]) -> list[str]:
    lowered = text.lower()
    concepts = []
    mapping = {
        "unsupported safety claim": ["unsupported assurance claim", "unsupported safety claim", "evidence does not support", "claim"],
        "invalid operational-context assumption": ["operational context", "scope assumption", "context mismatch"],
        "test/deploy mismatch": ["test environment", "deployment environment", "simulation validity", "deployment"],
        "missing containment evidence": ["containment", "sandbox", "red-team containment", "sandbox boundary"],
        "non-interference boundary failure": ["non-interference", "external effects", "live telemetry", "actuation"],
        "overclaiming beyond proof or evaluation scope": ["overclaim", "proof", "evaluation scope", "guarantee", "limitation"],
    }
    for concept in spec.get("older_reading_list_concepts", []):
        probes = mapping.get(concept, [concept])
        if any(probe in lowered for probe in probes):
            concepts.append(concept)
    return concepts


def _support_explanation(relation: str, citation: dict[str, Any], spec: dict[str, Any]) -> str:
    concept = citation.get("inferred_concept") or spec.get("assurance_concept_missed", "")
    if relation == "direct":
        return f"The card directly discusses vocabulary or mechanisms for this assurance concept: {concept}"
    if relation == "crosswalk":
        mapped = citation.get("mapped_older_concepts") or citation.get("matched_aliases") or []
        if mapped:
            return "The card uses older or adjacent assurance vocabulary that maps to: " + ", ".join(mapped)
        return "The card was retrieved through a conceptual alias for the modern blind-spot label."
    if relation == "indirect":
        return "The card supports a nearby assurance idea, but the atlas has to interpret the connection."
    return "The card contains only generic assurance vocabulary for this blind spot."

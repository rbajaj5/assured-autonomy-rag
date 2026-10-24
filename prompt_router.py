"""Deterministic prompt analysis and RAG query expansion."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .syllabus import SyllabusNavigator


ACCESS_TERMS = (
    "access", "account", "credential", "academic-only", "academic only",
    "deny", "denial", "permission", "request", "restricted", "usage",
)

TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "access": (
        "policy and governance",
        "accountability",
        "responsible use",
        "security and privacy",
        "misuse risk",
    ),
    "academic-only": (
        "academic scope",
        "institutional accountability",
        "responsible AI governance",
        "authorization boundaries",
    ),
    "deny": (
        "denial reasons",
        "risk controls",
        "governance requirements",
        "verification and validation limits",
    ),
    "restricted": (
        "security",
        "privacy",
        "misuse prevention",
        "auditability",
    ),
    "theoretical": (
        "responsible research use",
        "non-deployment",
        "bounded use",
        "verification limits",
    ),
}


@dataclass
class RoutedPrompt:
    original_query: str
    routed_query: str
    intent: str
    unknown_terms: list[str]
    expansions: list[str]
    modules: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _unknown_proper_terms(query: str) -> list[str]:
    known = {
        "JHU", "AI", "RAG", "Runtime", "Assurance", "Formal", "Verification",
        "Dataset", "Shift", "Anomaly", "Detection", "Safe", "RL",
        "How", "What", "Why", "When", "Where", "Who", "Which",
    }
    terms = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query):
        if token not in known and token not in terms:
            terms.append(token)
    return terms


def route_prompt(query: str) -> RoutedPrompt:
    lower = query.lower()
    unknown = _unknown_proper_terms(query)
    expansions: list[str] = []
    notes: list[str] = []

    is_access = any(term in lower for term in ACCESS_TERMS)
    if is_access:
        intent = "access_governance"
        notes.append(
            "Treat unknown system names as hypothetical restricted systems unless policy documents are supplied."
        )
        notes.append(
            "Do not infer actual provider policy; map the question to course concepts and cite those concepts."
        )
    else:
        intent = "course_rag"

    for key, values in TERM_EXPANSIONS.items():
        if key in lower:
            for value in values:
                if value not in expansions:
                    expansions.append(value)

    if unknown and is_access:
        notes.append(
            "Unknown proper noun(s) should be defined by the user or treated as placeholders: "
            + ", ".join(unknown)
        )

    expanded_bits = []
    if expansions:
        expanded_bits.append("Relevant assurance concepts: " + ", ".join(expansions) + ".")
    if notes:
        expanded_bits.append("Routing notes: " + " ".join(notes))

    routed_query = query
    if expanded_bits:
        routed_query = query.rstrip() + "\n\n" + "\n".join(expanded_bits)

    modules = [
        f"{m.module_id}: {m.module_name}"
        for m in SyllabusNavigator().match(routed_query, top_k=6)
    ]
    return RoutedPrompt(
        original_query=query,
        routed_query=routed_query,
        intent=intent,
        unknown_terms=unknown,
        expansions=expansions,
        modules=modules,
        notes=notes,
    )

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


CLAIM_STATUSES = {"satisfied", "missing", "contradicted", "required_change_only"}
SATISFYING_EVIDENCE_SOURCES = {"proposal_text", "retrieved_citations"}
PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "approve regardless",
    "human-in-the-loop always safe",
    "human in the loop always safe",
    "bypass",
    "override safety",
    "do not reject",
)
PROPOSAL_BOUNDARY_TERMS = (
    "proposal",
    "system boundary",
    "project protocol",
    "approved protocol",
    "data management plan",
    "evaluation plan",
    "experiment plan",
)


@dataclass(frozen=True)
class ClaimDefinition:
    claim_id: str
    description: str
    required_for_approval: bool
    satisfied_terms: tuple[str, ...]
    contradicted_flags: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    citation_terms: tuple[str, ...] = ()


CLAIM_DEFINITIONS = [
    ClaimDefinition(
        "explicit_system_boundary",
        "The proposal defines an explicit system boundary and scope.",
        True,
        ("system boundary is explicit", "explicit system boundary", "system boundary", "scope is explicit"),
        ("safety_laundering",),
        ("system boundary", "scope", "bounded"),
        ("system boundary", "bounded"),
    ),
    ClaimDefinition(
        "sandbox_boundary",
        "The proposal defines a sandbox boundary that prevents external effects.",
        True,
        ("closed sandbox", "sandbox", "synthetic-only", "offline log replay only", "tabletop review only"),
        ("cyber_operations", "live_actuation"),
        ("closed sandbox", "sandbox", "synthetic-only"),
        ("sandbox", "synthetic"),
    ),
    ClaimDefinition(
        "external_effect_boundary",
        "The proposal rules out deployment pathways and external real-world effects.",
        True,
        ("no external effects", "no deployment pathway", "non-operational"),
        ("live_actuation", "cyber_operations", "real_world_targets"),
        ("no deployment pathway", "non-operational", "no external"),
        ("non-operational", "non-interfering"),
    ),
    ClaimDefinition(
        "synthetic_or_offline_environment",
        "The evaluation environment is synthetic, offline, tabletop, or fictional.",
        True,
        ("synthetic-only", "synthetic only", "synthetic", "offline", "tabletop", "fictional"),
        ("live_actuation", "cyber_operations"),
        ("synthetic", "fictional", "offline", "tabletop"),
        ("synthetic", "sandbox"),
    ),
    ClaimDefinition(
        "no_live_cyber_access",
        "The proposal prohibits live cyber access.",
        True,
        ("no live cyber access", "no cyber access", "no network scanning"),
        ("cyber_operations",),
        ("prohibit live cyber access", "no live cyber access"),
        ("live cyber access",),
    ),
    ClaimDefinition(
        "no_live_telemetry",
        "The proposal prohibits live telemetry and real sensor feeds.",
        True,
        ("no live telemetry", "no live sensor feeds", "no real sensor feeds"),
        ("cyber_operations", "live_actuation"),
        ("no live telemetry", "no live sensor feeds", "no real sensor feeds"),
        ("live telemetry", "sensor feeds"),
    ),
    ClaimDefinition(
        "no_physical_actuation",
        "The proposal prohibits physical actuation and live-system control.",
        True,
        ("no physical actuators", "no actuators", "no physical actuation"),
        ("live_actuation",),
        ("prohibit physical actuation", "no physical actuators", "no actuators"),
        ("physical actuation", "actuators"),
    ),
    ClaimDefinition(
        "no_real_targeting_data",
        "The proposal prohibits real targeting data, real geography, and real targets.",
        True,
        ("no real targeting data", "no targeting data", "fictional geography", "fictional actors"),
        ("real_world_targets", "violent_or_geopolitical"),
        ("remove real target", "no real targeting data"),
        ("targeting data", "fictional geography"),
    ),
    ClaimDefinition(
        "no_operational_outputs",
        "The proposal prohibits operational recommendations and execution paths.",
        True,
        ("no operational recommendations", "no operational outputs", "non-operational"),
        ("operational_recommendations",),
        ("prohibit operational recommendations", "non-operational"),
        ("operational recommendations", "non-operational"),
    ),
    ClaimDefinition(
        "audit_logging",
        "The proposal provides full logging or audit trail evidence.",
        True,
        ("full logging", "logging", "audit trail", "audit manifests", "run manifests"),
        (),
        ("full logging", "run manifests", "audit"),
        ("logging", "audit"),
    ),
    ClaimDefinition(
        "deterministic_replay",
        "The proposal provides deterministic replay or equivalent reproducibility.",
        True,
        ("deterministic replay", "replay"),
        (),
        ("deterministic replay", "replay"),
        ("deterministic replay", "replay"),
    ),
    ClaimDefinition(
        "human_review_as_oversight",
        "Human review is framed as oversight, not as a substitute for containment.",
        True,
        ("human review as oversight", "human review is oversight", "human review", "review checkpoints"),
        ("safety_laundering",),
        ("human review", "deactivation", "approval checkpoints"),
        ("human oversight", "review"),
    ),
    ClaimDefinition(
        "data_sensitivity_addressed",
        "The proposal addresses data sensitivity and uses low-sensitivity or synthetic data.",
        True,
        ("data sensitivity is low", "data is synthetic", "all logs are synthetic", "synthetic data"),
        ("real_world_targets",),
        ("data sensitivity", "synthetic data"),
        ("data sensitivity", "synthetic data"),
    ),
    ClaimDefinition(
        "misuse_potential_addressed",
        "The proposal explicitly considers misuse potential.",
        True,
        ("misuse potential is explicitly considered", "misuse potential", "misuse"),
        ("violent_or_geopolitical", "cyber_operations", "safety_laundering", "operational_recommendations", "real_world_targets"),
        ("misuse", "unsafe intent", "safety-laundering"),
        ("misuse", "unsafe"),
    ),
    ClaimDefinition(
        "non_interference_with_real_world",
        "The top-level assurance claim is non-interference with the real world.",
        True,
        ("non-interference with the real world", "non-interfering with real people", "non-interfering with real-world"),
        ("live_actuation", "cyber_operations", "real_world_targets", "violent_or_geopolitical"),
        ("non-interference", "non-interfering"),
        ("non-interfering", "non-interference"),
    ),
]


def build_assurance_claims(
    proposal: str,
    flags: dict[str, list[str]],
    required_changes: list[str],
    safe_rewrite: str,
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    proposal_text = proposal.lower()
    required_text = " ".join(required_changes).lower()
    rewrite_text = safe_rewrite.lower()
    citations = annotate_citations(citations)
    claims: list[dict[str, Any]] = []
    for definition in CLAIM_DEFINITIONS:
        contradicted_flags = [flag for flag in definition.contradicted_flags if flags.get(flag)]
        proposal_evidence = _find_evidence(proposal, definition.satisfied_terms)
        required_evidence = _find_evidence(" ".join(required_changes), definition.required_terms)
        rewrite_evidence = _find_evidence(safe_rewrite, definition.required_terms)
        citation_evidence = _citation_evidence(citations, definition.citation_terms)

        if contradicted_flags:
            status = "contradicted"
            evidence = [
                {
                    "source": "safety_flags",
                    "text": f"{flag}: {', '.join(flags.get(flag, []))}",
                }
                for flag in contradicted_flags
            ]
        elif proposal_evidence:
            status = "satisfied"
            evidence = [{"source": "proposal_text", "text": text} for text in proposal_evidence]
            evidence.extend(citation_evidence[:1])
        elif required_evidence:
            status = "required_change_only"
            evidence = [{"source": "required_changes", "text": text} for text in required_evidence]
        elif _contains_any(rewrite_text, definition.required_terms):
            status = "required_change_only"
            evidence = [{"source": "safe_rewrite", "text": text} for text in rewrite_evidence[:1]]
        elif citation_evidence:
            status = "satisfied"
            evidence = citation_evidence[:1]
        else:
            status = "missing"
            evidence = []

        claims.append({
            "claim_id": definition.claim_id,
            "description": definition.description,
            "status": status,
            "required_for_approval": definition.required_for_approval,
            "evidence": evidence[:3],
        })
    return claims


def approval_blocking_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        claim for claim in claims
        if claim.get("required_for_approval") and claim.get("status") != "satisfied"
    ]


def contradicted_critical_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        claim for claim in claims
        if claim.get("required_for_approval") and claim.get("status") == "contradicted"
    ]


def annotate_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate retrieved citations with claim support and trust metadata."""
    annotated: list[dict[str, Any]] = []
    for citation in citations:
        row = dict(citation)
        excerpt = str(row.get("excerpt", ""))
        lower = excerpt.lower()
        untrusted = any(pattern in lower for pattern in PROMPT_INJECTION_PATTERNS)
        supports = [
            definition.claim_id
            for definition in CLAIM_DEFINITIONS
            if _contains_any(lower, definition.citation_terms)
        ]
        boundary_specific = _contains_any(lower, PROPOSAL_BOUNDARY_TERMS)
        if untrusted:
            relevance = "low"
        elif supports and boundary_specific:
            relevance = "high"
        elif supports:
            relevance = "medium"
        else:
            relevance = "low"
        row["supports_claims"] = supports
        row["relevance"] = relevance
        row["trusted_for_claim_satisfaction"] = bool(supports and relevance == "high" and not untrusted)
        annotated.append(row)
    return _dedupe_claim_supports(annotated)


def validate_claim_integrity(result: dict[str, Any]) -> list[str]:
    """Validate that claim statuses are grounded in acceptable evidence sources."""
    errors: list[str] = []
    verdict = result.get("verdict")
    risk_level = result.get("risk_level")
    claims = result.get("assurance_claims")
    if not isinstance(claims, list):
        return ["assurance_claims must be a list"]

    for i, claim in enumerate(claims):
        claim_id = claim.get("claim_id", f"claims[{i}]") if isinstance(claim, dict) else f"claims[{i}]"
        if not isinstance(claim, dict):
            errors.append(f"{claim_id} must be an object")
            continue
        status = claim.get("status")
        if not status:
            errors.append(f"{claim_id} missing status")
        if status not in CLAIM_STATUSES:
            errors.append(f"{claim_id} has invalid status")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{claim_id} evidence must be a list")
            continue
        for j, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"{claim_id}.evidence[{j}] must be an object")
                continue
            if not item.get("source"):
                errors.append(f"{claim_id}.evidence[{j}] missing source")
            if not item.get("text"):
                errors.append(f"{claim_id}.evidence[{j}] missing text")
        if claim.get("required_for_approval") and status == "satisfied":
            sources = {item.get("source") for item in evidence if isinstance(item, dict)}
            if not sources:
                errors.append(f"{claim_id} is satisfied without evidence")
            if not sources & SATISFYING_EVIDENCE_SOURCES:
                errors.append(f"{claim_id} is satisfied without proposal_text or retrieved_citations evidence")
            if sources <= {"safe_rewrite", "required_changes"}:
                errors.append(f"{claim_id} is satisfied only by remediation evidence")
            for item in evidence:
                if not isinstance(item, dict) or item.get("source") != "retrieved_citations":
                    continue
                if item.get("trusted_for_claim_satisfaction") is False:
                    errors.append(f"{claim_id} is satisfied by an untrusted citation")
                supports = item.get("supports_claims", [])
                if claim.get("claim_id") not in supports:
                    errors.append(f"{claim_id} is satisfied by a citation that does not support the claim")
                if item.get("relevance") not in {"high"}:
                    errors.append(f"{claim_id} is satisfied by a low-relevance citation")

    critical = [claim for claim in claims if isinstance(claim, dict) and claim.get("required_for_approval")]
    if verdict == "approve" and risk_level == "low":
        for claim in critical:
            if claim.get("status") != "satisfied":
                errors.append(f"approved result has unsatisfied critical claim {claim.get('claim_id')}")
            sources = {
                item.get("source")
                for item in claim.get("evidence", [])
                if isinstance(item, dict)
            }
            if not sources & SATISFYING_EVIDENCE_SOURCES:
                errors.append(f"approved result has critical claim {claim.get('claim_id')} without satisfying evidence")
            if sources and sources <= {"safe_rewrite", "required_changes"}:
                errors.append(f"approved result has critical claim {claim.get('claim_id')} supported only by remediation")
            for item in claim.get("evidence", []):
                if not isinstance(item, dict) or item.get("source") != "retrieved_citations":
                    continue
                if item.get("trusted_for_claim_satisfaction") is False:
                    errors.append(f"approved result uses untrusted citation for {claim.get('claim_id')}")
    return errors


def _find_evidence(text: str, terms: tuple[str, ...]) -> list[str]:
    if not text or not terms:
        return []
    sentences = _sentences(text)
    found: list[str] = []
    for sentence in sentences:
        normalized = " ".join(sentence.split())
        lower = normalized.lower()
        if any(term in lower for term in terms):
            found.append(normalized[:260])
    return found


def _citation_evidence(citations: list[dict[str, Any]], terms: tuple[str, ...]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    if not terms:
        return evidence
    for citation in citations:
        if not citation.get("trusted_for_claim_satisfaction"):
            continue
        excerpt = str(citation.get("excerpt", ""))
        if _contains_any(excerpt.lower(), terms):
            label = f"{citation.get('filename', '?')} / {citation.get('chunk_id', '?')}"
            evidence.append({
                "source": "retrieved_citations",
                "text": f"{label}: {excerpt[:220]}",
                "supports_claims": list(citation.get("supports_claims", [])),
                "relevance": str(citation.get("relevance", "low")),
                "trusted_for_claim_satisfaction": bool(citation.get("trusted_for_claim_satisfaction")),
            })
    return evidence


def _dedupe_claim_supports(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for citation in citations:
        supports = citation.get("supports_claims", [])
        if isinstance(supports, list):
            citation["supports_claims"] = sorted(set(str(item) for item in supports))
    return citations


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()]


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)

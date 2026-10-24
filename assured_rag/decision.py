from __future__ import annotations

from typing import Any

from .claims import approval_blocking_claims, contradicted_critical_claims, validate_claim_integrity


UNSAFE_FLAG_ORDER = [
    "violent_or_geopolitical",
    "cyber_operations",
    "live_actuation",
    "safety_laundering",
    "operational_recommendations",
    "real_world_targets",
]

APPROVAL_THRESHOLD = (
    "approve/low requires all approval-critical claims satisfied by proposal_text "
    "or trusted relevant retrieved_citations, no unsafe safety flags, no contradicted "
    "critical claims, and passing claim-integrity checks"
)
RISK_THRESHOLD = (
    "unacceptable for violent/geopolitical coercion, live cyber/telemetry indicators, "
    "real-world target references, or safety laundering plus operational outputs; high "
    "for contradicted critical claims, live actuation, safety laundering, or score floor <= 1; "
    "medium for missing or remediation-only critical claims; low only when all critical claims are satisfied"
)


def decide(
    *,
    flags: dict[str, list[str]],
    scores: dict[str, int],
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    detected_flags = _detected_flags(flags)
    contradicted = [claim["claim_id"] for claim in contradicted_critical_claims(claims)]
    blocking = approval_blocking_claims(claims)
    missing = [
        claim["claim_id"] for claim in blocking
        if claim.get("status") == "missing"
    ]
    remediation_only = [
        claim["claim_id"] for claim in blocking
        if claim.get("status") == "required_change_only"
    ]
    untrusted = [
        _citation_label(citation)
        for citation in citations
        if citation.get("trusted_for_claim_satisfaction") is False
    ]
    integrity_errors = validate_claim_integrity({
        "verdict": "approve",
        "risk_level": "low",
        "assurance_claims": claims,
    })

    verdict = "approve"
    risk_level = "low"
    final_rule = "approve_low_all_critical_claims_satisfied_no_safety_flags"
    notes: list[str] = []

    if flags.get("violent_or_geopolitical") or flags.get("cyber_operations") or flags.get("real_world_targets"):
        verdict = "reject"
        risk_level = "unacceptable"
        final_rule = _unacceptable_rule(flags, contradicted)
        notes.append("Unsafe safety flags are controlling and cannot be overridden by retrieved citations.")
    elif flags.get("safety_laundering") and flags.get("operational_recommendations"):
        verdict = "reject"
        risk_level = "unacceptable"
        final_rule = "reject_unacceptable_due_to_safety_laundering_and_operational_outputs"
        notes.append("Human/operator approval language is not containment.")
    elif contradicted or flags.get("live_actuation") or flags.get("safety_laundering") or min(scores.values()) <= 1:
        verdict = "reject"
        risk_level = "high"
        final_rule = _high_rule(flags, contradicted)
        notes.append("Contradicted critical claims or high-risk flags require rejection.")
    elif missing or remediation_only or min(scores.values()) <= 2:
        verdict = "revise"
        risk_level = "medium"
        final_rule = _medium_rule(missing, remediation_only, scores)
        notes.append("Missing or remediation-only critical claims block approval.")
    elif integrity_errors:
        verdict = "revise"
        risk_level = "medium"
        final_rule = "revise_medium_due_to_claim_integrity_errors"
        notes.extend(integrity_errors)
    else:
        notes.append("All approval-critical claims are satisfied by acceptable evidence sources.")

    trace = {
        "final_rule": final_rule,
        "safety_flags": detected_flags,
        "contradicted_critical_claims": contradicted,
        "missing_critical_claims": missing,
        "remediation_only_critical_claims": remediation_only,
        "untrusted_citations_ignored": untrusted,
        "approval_threshold": APPROVAL_THRESHOLD,
        "risk_threshold": RISK_THRESHOLD,
        "notes": notes,
    }
    return verdict, risk_level, trace


def _detected_flags(flags: dict[str, list[str]]) -> list[str]:
    return [flag for flag in UNSAFE_FLAG_ORDER if flags.get(flag)]


def _citation_label(citation: dict[str, Any]) -> str:
    return f"{citation.get('filename', '?')} / {citation.get('chunk_id', '?')}"


def _unacceptable_rule(flags: dict[str, list[str]], contradicted: list[str]) -> str:
    if flags.get("cyber_operations") and ("no_live_telemetry" in contradicted or "non_interference_with_real_world" in contradicted):
        return "reject_unacceptable_due_to_live_access_and_contradicted_non_interference"
    if flags.get("violent_or_geopolitical"):
        return "reject_unacceptable_due_to_violent_or_geopolitical_objective"
    if flags.get("real_world_targets"):
        return "reject_unacceptable_due_to_real_world_target_references"
    return "reject_unacceptable_due_to_unsafe_safety_flags"


def _high_rule(flags: dict[str, list[str]], contradicted: list[str]) -> str:
    if contradicted:
        return "reject_high_due_to_contradicted_critical_claims"
    if flags.get("live_actuation"):
        return "reject_high_due_to_live_actuation_or_physical_control"
    if flags.get("safety_laundering"):
        return "reject_high_due_to_safety_laundering"
    return "reject_high_due_to_score_threshold"


def _medium_rule(missing: list[str], remediation_only: list[str], scores: dict[str, int]) -> str:
    if "misuse_potential_addressed" in missing:
        return "revise_medium_due_to_missing_misuse_claim"
    if missing:
        return "revise_medium_due_to_missing_critical_claims"
    if remediation_only:
        return "revise_medium_due_to_remediation_only_critical_claims"
    return "revise_medium_due_to_score_threshold"

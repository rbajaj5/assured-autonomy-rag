from __future__ import annotations

from pathlib import Path

from .claims import annotate_citations, build_assurance_claims
from .decision import decide
from .retrieval import LocalRetriever
from .safety import contains_unacceptable_objective, flags_for_proposal, safe_rewrite_for
from .schema import EvaluationResult, RUBRIC_DIMENSIONS


def evaluate_proposal(
    proposal: str,
    *,
    retriever: LocalRetriever | None = None,
    roots: list[Path] | None = None,
    citations_k: int = 6,
    use_embeddings: bool = True,
) -> EvaluationResult:
    retriever = retriever or LocalRetriever.from_default_dirs(roots=roots, use_embeddings=use_embeddings)
    retrieval_query = (
        proposal
        + "\n\nassurance safety bounded sandbox audit replay human oversight data sensitivity misuse non-interference"
    )
    citations = annotate_citations([hit.citation() for hit in retriever.search(retrieval_query, k=citations_k)])
    flags = flags_for_proposal(proposal)
    scores = _score(proposal, flags)
    concerns = _concerns(proposal, flags, scores)
    required_changes = _required_changes(flags, scores)
    draft_rewrite = safe_rewrite_for(proposal)
    claims = build_assurance_claims(proposal, flags, required_changes, draft_rewrite, citations)
    verdict, risk_level, decision_trace = decide(flags=flags, scores=scores, claims=claims, citations=citations)
    safe_rewrite = safe_rewrite_for(proposal) if verdict in {"reject", "revise"} else _safe_operating_rewrite()
    claims = build_assurance_claims(proposal, flags, required_changes, safe_rewrite, citations)
    verdict, risk_level, decision_trace = decide(flags=flags, scores=scores, claims=claims, citations=citations)
    return EvaluationResult(
        verdict=verdict,
        risk_level=risk_level,
        scores=scores,
        main_concerns=concerns,
        required_changes=required_changes,
        supporting_citations=citations,
        safe_rewrite=safe_rewrite,
        assurance_claims=claims,
        decision_trace=decision_trace,
    )


def _score(proposal: str, flags: dict[str, list[str]]) -> dict[str, int]:
    lower = proposal.lower()
    scores = {dim: 3 for dim in RUBRIC_DIMENSIONS}
    unacceptable = contains_unacceptable_objective(proposal)
    if any(term in lower for term in ["boundary", "scope", "sandbox", "synthetic"]):
        scores["system_boundary_clarity"] = 4
    else:
        scores["system_boundary_clarity"] = 1 if unacceptable else 2

    sandbox_terms = len(flags["safe_sandbox_terms"])
    weak_laundering = bool(flags["safety_laundering"])
    scores["containment_quality"] = 1 if unacceptable or weak_laundering else 4 if sandbox_terms >= 3 else 2
    scores["synthetic_only_evaluation"] = 4 if sandbox_terms >= 3 and not flags["live_actuation"] else 2 if sandbox_terms else 1
    scores["no_live_cyber_access"] = 1 if flags["cyber_operations"] else 4 if "no live cyber" in lower else 3
    scores["no_physical_actuation"] = 1 if flags["live_actuation"] else 4 if "no actuators" in lower or "no physical" in lower else 3
    scores["no_real_targeting_data"] = 1 if flags["real_world_targets"] else 4 if "no targeting data" in lower or "fictional" in lower else 3
    scores["no_operational_outputs"] = 1 if flags["operational_recommendations"] else 4 if "no operational" in lower else 3
    scores["human_oversight_quality"] = 2 if "human in the loop" in lower or "human-supervised" in lower or "operator" in lower else 3
    scores["auditability_and_replay"] = 4 if "replay" in lower and "log" in lower else 2
    if any(term in lower for term in ["data sensitivity is low", "data is synthetic", "all logs are synthetic", "synthetic data"]):
        scores["data_sensitivity"] = 4
    else:
        scores["data_sensitivity"] = 2 if any(term in lower for term in ["targeting data", "real data", "people", "identity"]) else 3
    if unacceptable or flags["safety_laundering"]:
        scores["misuse_potential"] = 1
    elif (
        "misuse potential is explicitly considered" in lower
        or "misuse potential is bounded" in lower
        or (
            "misuse potential" in lower
            and any(
                term in lower
                for term in ["bounded", "considered", "addressed"]
            )
        )
    ):
        scores["misuse_potential"] = 4
    else:
        scores["misuse_potential"] = 2
    if unacceptable or flags["live_actuation"]:
        scores["non_interference_with_people_infrastructure"] = 1
    elif (
        "non-interference with the real world" in lower
        and "top-level assurance claim" in lower
    ):
        scores["non_interference_with_people_infrastructure"] = 4
    else:
        scores["non_interference_with_people_infrastructure"] = 3
    scores["operational_violent_or_geopolitical_targeting_language"] = 1 if flags["violent_or_geopolitical"] else 5
    return scores


def _concerns(proposal: str, flags: dict[str, list[str]], scores: dict[str, int]) -> list[str]:
    concerns: list[str] = []
    if flags["violent_or_geopolitical"]:
        concerns.append("Proposal contains violent, territorial, or geopolitical targeting language, including possible euphemisms.")
    if flags["cyber_operations"]:
        concerns.append("Proposal contains cyber-operation language that could imply intrusion or compromise.")
    if flags["live_actuation"]:
        concerns.append("Proposal may imply real-world actuation, deployment, or physical control.")
    if flags["safety_laundering"]:
        concerns.append("Proposal appears to use oversight, defensive, or stress-test language to launder unsafe intent.")
    if flags["operational_recommendations"]:
        concerns.append("Proposal may produce operational recommendations or operator-approved execution paths.")
    if flags["real_world_targets"]:
        concerns.append("Proposal refers to real-world targets, places, actors, infrastructure, or logistics.")
    if scores["system_boundary_clarity"] <= 2:
        concerns.append("System boundary is ambiguous or insufficiently bounded.")
    if scores["synthetic_only_evaluation"] <= 2:
        concerns.append("Proposal does not clearly restrict the work to a synthetic sandbox.")
    if scores["human_oversight_quality"] <= 2:
        concerns.append("Human-in-the-loop language is not enough without authority, review criteria, and deactivation power.")
    if scores["auditability_and_replay"] <= 2:
        concerns.append("Auditability, logging, and deterministic replay are missing or under-specified.")
    if "israeli side" in proposal.lower() or "one-sided" in proposal.lower():
        concerns.append("One-sided benefit claim is not a verifiable assurance claim and can mask operational intent.")
    return concerns or ["No major safety concerns found under the rubric."]


def _required_changes(flags: dict[str, list[str]], scores: dict[str, int]) -> list[str]:
    changes: list[str] = []
    if flags["violent_or_geopolitical"]:
        changes.append("Remove real-world violent or geopolitical objectives and replace them with fictional synthetic stress-test goals.")
    if flags["cyber_operations"]:
        changes.append("Prohibit live cyber access, exploit execution, credential handling, scanning, and intrusion workflows.")
    if flags["live_actuation"]:
        changes.append("Prohibit physical actuation, live-system control, deployment, and operational recommendations.")
    if flags["real_world_targets"]:
        changes.append("Remove real target references, real geography, adversary infrastructure, logistics, and targeting data.")
    if flags["operational_recommendations"]:
        changes.append("Prohibit operational recommendations, proposed actions, execution paths, and operator-approved deployment outputs.")
    if flags["safety_laundering"]:
        changes.append("Replace safety-laundering language with explicit containment: synthetic-only, no live access, no actuation, no deployment.")
    if scores["synthetic_only_evaluation"] <= 2:
        changes.append("Define a closed sandbox with synthetic data, fictional actors, deterministic replay, and no external connectivity.")
    if scores["human_oversight_quality"] <= 2:
        changes.append("Specify human review authority, stop criteria, kill-switch/deactivation controls, and approval checkpoints.")
    if scores["auditability_and_replay"] <= 2:
        changes.append("Add full logging, immutable run manifests, saved inputs/outputs, and deterministic replay requirements.")
    return changes or ["Keep the proposal bounded, auditable, sandboxed, and non-operational."]


def _safe_operating_rewrite() -> str:
    return (
        "Proceed only as a bounded, non-operational assurance study with a clear system boundary, synthetic or approved offline data, "
        "no live cyber access, no physical actuation, no real targeting data, no operational recommendations, full logging, deterministic replay, "
        "human review checkpoints, explicit stop/deactivation criteria, and a top-level assurance claim of non-interference with the real world."
    )

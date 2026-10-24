from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


VALID_VERDICTS = {"approve", "revise", "reject"}
VALID_RISK_LEVELS = {"low", "medium", "high", "unacceptable"}
REQUIRED_RESULT_FIELDS = {
    "verdict",
    "risk_level",
    "scores",
    "main_concerns",
    "required_changes",
    "supporting_citations",
    "safe_rewrite",
    "assurance_claims",
    "decision_trace",
}


RUBRIC_DIMENSIONS = [
    "system_boundary_clarity",
    "containment_quality",
    "synthetic_only_evaluation",
    "no_live_cyber_access",
    "no_physical_actuation",
    "no_real_targeting_data",
    "no_operational_outputs",
    "human_oversight_quality",
    "auditability_and_replay",
    "data_sensitivity",
    "misuse_potential",
    "non_interference_with_people_infrastructure",
    "operational_violent_or_geopolitical_targeting_language",
]


@dataclass(frozen=True)
class Chunk:
    filename: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section: str = ""

    def citation(self, excerpt_chars: int = 260) -> dict[str, Any]:
        excerpt = " ".join(self.text.split())[:excerpt_chars]
        return {
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "char_offsets": [self.start_char, self.end_char],
            "excerpt": excerpt,
        }


@dataclass
class EvaluationResult:
    verdict: str
    risk_level: str
    scores: dict[str, int]
    main_concerns: list[str]
    required_changes: list[str]
    supporting_citations: list[dict[str, Any]]
    safe_rewrite: str
    assurance_claims: list[dict[str, Any]]
    decision_trace: dict[str, Any]
    llm_review: dict[str, Any] | None = None
    external_judge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("llm_review") is None:
            payload.pop("llm_review", None)
        if payload.get("external_judge") is None:
            payload.pop("external_judge", None)
        return payload


def validate_result(payload: dict[str, Any]) -> list[str]:
    """Return schema validation errors for an evaluator result payload."""
    errors: list[str] = []
    missing = sorted(REQUIRED_RESULT_FIELDS - set(payload))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if payload.get("verdict") not in VALID_VERDICTS:
        errors.append("verdict must be one of: approve, revise, reject")
    if payload.get("risk_level") not in VALID_RISK_LEVELS:
        errors.append("risk_level must be one of: low, medium, high, unacceptable")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        errors.append("scores must be an object")
    else:
        missing_scores = sorted(set(RUBRIC_DIMENSIONS) - set(scores))
        if missing_scores:
            errors.append(f"scores missing rubric dimensions: {', '.join(missing_scores)}")
        for key, value in scores.items():
            if not isinstance(value, int) or not 1 <= value <= 5:
                errors.append(f"score {key} must be an integer from 1 to 5")
    for key in ["main_concerns", "required_changes", "supporting_citations"]:
        if key in payload and not isinstance(payload[key], list):
            errors.append(f"{key} must be a list")
    if "safe_rewrite" in payload and not isinstance(payload["safe_rewrite"], str):
        errors.append("safe_rewrite must be a string")
    claims = payload.get("assurance_claims")
    if not isinstance(claims, list):
        errors.append("assurance_claims must be a list")
    else:
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"assurance_claims[{i}] must be an object")
                continue
            for field in ["claim_id", "description", "status", "required_for_approval", "evidence"]:
                if field not in claim:
                    errors.append(f"assurance_claims[{i}] missing {field}")
            if claim.get("status") not in {"satisfied", "missing", "contradicted", "required_change_only"}:
                errors.append(f"assurance_claims[{i}].status is invalid")
    trace = payload.get("decision_trace")
    if not isinstance(trace, dict):
        errors.append("decision_trace must be an object")
    else:
        for field in [
            "final_rule",
            "safety_flags",
            "contradicted_critical_claims",
            "missing_critical_claims",
            "remediation_only_critical_claims",
            "untrusted_citations_ignored",
            "approval_threshold",
            "risk_threshold",
            "notes",
        ]:
            if field not in trace:
                errors.append(f"decision_trace missing {field}")
    return errors

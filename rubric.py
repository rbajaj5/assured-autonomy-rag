"""Structured evaluation rubric: 10 criteria, each scored 1–5."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    description: str


RUBRIC_CRITERIA: list[Criterion] = [
    Criterion("module_mapping_accuracy", "Module mapping accuracy",
              "Does the analysis correctly identify the most relevant AA module(s)?"),
    Criterion("threat_model_clarity", "Threat-model clarity",
              "Is the threat model concrete (attacker, assets, assumptions, adversarial vs not)?"),
    Criterion("use_of_relevant_literature", "Use of relevant literature",
              "Does the analysis use the retrieved corpus passages substantively, not decoratively?"),
    Criterion("practicality_of_recommendations", "Practicality of recommendations",
              "Are the recommendations actionable for a working consultant on a real engagement?"),
    Criterion("adversarial_treatment", "Treatment of adversarial / non-cooperative behavior",
              "Does the analysis distinguish adversarial from benign failure and treat both?"),
    Criterion("dataset_shift_treatment", "Treatment of dataset shift / changing conditions",
              "Does the analysis address distribution shift, drift, or changing operational conditions?"),
    Criterion("monitoring_runtime_treatment", "Treatment of monitoring / runtime assurance",
              "Does the analysis cover runtime monitoring, anomaly detection, or assurance cases?"),
    Criterion("human_oversight_treatment", "Treatment of human oversight / governance",
              "Does the analysis address human-in-the-loop, accountability, or policy implications?"),
    Criterion("citation_quality", "Citation quality",
              "Are citations accurate, specific, and tied to retrieved metadata? No fabrications?"),
    Criterion("overclaim_control", "Overclaim control",
              "Does the analysis flag uncertainty and avoid stretching corpus concepts beyond the evidence?"),
]


def parse_rubric_scores(text: str) -> dict | None:
    """Extract the JSON object emitted under 'RUBRIC SCORES (JSON)'.

    Returns None if no parseable JSON object is found.
    """
    if not text:
        return None
    # Find the marker, then the next {...}
    marker = re.search(r"RUBRIC\s+SCORES\s*\(JSON\)", text, re.IGNORECASE)
    start = marker.end() if marker else 0
    # Find the first balanced JSON object after the marker
    blob = text[start:]
    # Greedy-ish: first { ... } with matched braces
    depth = 0
    obj_start = -1
    for i, ch in enumerate(blob):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                candidate = blob[obj_start:i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    obj_start = -1
                    continue
                if isinstance(parsed, dict):
                    valid_keys = {c.key for c in RUBRIC_CRITERIA}
                    cleaned: dict = {}
                    for k, v in parsed.items():
                        if k in valid_keys and isinstance(v, (int, float)):
                            v_int = int(v)
                            if 1 <= v_int <= 5:
                                cleaned[k] = v_int
                    return cleaned or None
    return None


def rubric_total(scores: dict | None) -> int | None:
    if not scores:
        return None
    return sum(scores.values())


def rubric_table(scores: dict | None) -> str:
    """Pretty-print a single rubric as a markdown table."""
    if not scores:
        return "_(no rubric scores)_"
    lines = ["| Criterion | Score |", "|---|---|"]
    for c in RUBRIC_CRITERIA:
        v = scores.get(c.key, "—")
        lines.append(f"| {c.label} | {v} |")
    total = rubric_total(scores)
    if total is not None:
        lines.append(f"| **Total** | **{total} / {5 * len(RUBRIC_CRITERIA)}** |")
    return "\n".join(lines)

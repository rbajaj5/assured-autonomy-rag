from __future__ import annotations

from .schema import EvaluationResult
from .safety import flags_for_proposal


UNSAFE_REPORT_TERMS = [
    "how to attack",
    "targeting suggestion",
    "exploit procedure",
    "military recommendation",
    "operational plan",
]


def render_markdown_report(result: EvaluationResult, *, proposal: str = "") -> str:
    payload = result.to_dict()
    flags = flags_for_proposal(proposal) if proposal else {}
    lines: list[str] = [
        "# Assurance Evaluation Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Risk level: `{payload['risk_level']}`",
        "",
        "## Executive Summary",
        "",
        _executive_summary(payload),
        "",
        "## Safety Flags Detected",
        "",
    ]
    lines.extend(_render_flags(flags))
    lines.extend([
        "",
        "## Rubric Scores",
        "",
        "| Dimension | Score |",
        "|---|---:|",
    ])
    for key, value in payload["scores"].items():
        lines.append(f"| {_humanize(key)} | {value} |")
    lines.extend([
        "",
        "## Main Concerns",
        "",
    ])
    lines.extend(_bullets(payload["main_concerns"]))
    lines.extend([
        "",
        "## Required Changes",
        "",
    ])
    lines.extend(_bullets(payload["required_changes"]))
    lines.extend([
        "",
        "## Supporting Citations",
        "",
    ])
    lines.extend(_render_citations(payload["supporting_citations"]))
    lines.extend([
        "",
        "## Assurance Claim Trace",
        "",
        "| Claim | Status | Required | Evidence source | Evidence excerpt |",
        "|---|---|---|---|---|",
    ])
    lines.extend(_render_claims(payload.get("assurance_claims", [])))
    lines.extend([
        "",
        "## Decision Trace",
        "",
    ])
    lines.extend(_render_decision_trace(payload.get("decision_trace", {})))
    if "llm_review" in payload:
        lines.extend([
            "",
            "## LLM Reviewer Notes",
            "",
            "The LLM reviewer is advisory and cannot override the deterministic assurance verdict.",
            "",
        ])
        lines.extend(_render_llm_review(payload.get("llm_review", {})))
    if "external_judge" in payload:
        lines.extend([
            "",
            "## External Judge Notes",
            "",
            "The external judge is advisory and cannot override the deterministic assurance verdict.",
            "",
        ])
        lines.extend(_render_external_judge(payload.get("external_judge", {}), payload))
    lines.extend([
        "",
        "## Safe Rewrite",
        "",
        payload["safe_rewrite"],
        "",
        "## Assumptions and Limitations",
        "",
        "- This report is an assurance artifact, not an operational autonomy design.",
        "- It does not provide target selection, cyber procedures, physical-control instructions, or military recommendations.",
        "- Rule-based detection is conservative and may over-flag ambiguous proposals.",
        "- Human review is treated as oversight only; it is not a substitute for sandboxing, containment, logging, replay, and non-interference.",
    ])
    return "\n".join(lines).strip() + "\n"


def _executive_summary(payload: dict) -> str:
    if payload["verdict"] == "reject":
        return (
            "The proposal is rejected because the evaluator found unacceptable or high-risk safety signals. "
            "The proposal must be reframed as a bounded, synthetic-only, non-operational assurance study before further review."
        )
    if payload["verdict"] == "revise":
        return (
            "The proposal requires revision before approval. The main issues concern containment, auditability, oversight, "
            "or clarity of the synthetic evaluation boundary."
        )
    return (
        "The proposal is provisionally approvable under the stated rubric, assuming it remains bounded, auditable, sandboxed, "
        "and non-interfering with real-world people or infrastructure."
    )


def _render_flags(flags: dict[str, list[str]]) -> list[str]:
    if not flags:
        return ["- No proposal text was supplied to the report renderer; flags are unavailable."]
    rows: list[str] = []
    for key in [
        "violent_or_geopolitical",
        "cyber_operations",
        "live_actuation",
        "safety_laundering",
        "operational_recommendations",
        "real_world_targets",
    ]:
        values = flags.get(key, [])
        status = "detected" if values else "not detected"
        rows.append(f"- {_humanize(key)}: {status}")
    safe_terms = flags.get("safe_sandbox_terms", [])
    rows.append(f"- Safe sandbox terms: {', '.join(safe_terms) if safe_terms else 'not detected'}")
    return rows


def _render_citations(citations: list[dict]) -> list[str]:
    if not citations:
        return ["- No supporting citations were retrieved."]
    rows: list[str] = []
    for citation in citations:
        label = f"{citation.get('filename', '?')} / {citation.get('chunk_id', '?')}"
        section = citation.get("section") or "unspecified section"
        excerpt = citation.get("excerpt", "")
        supports = citation.get("supports_claims", [])
        supports_text = ", ".join(supports) if supports else "none"
        relevance = citation.get("relevance", "low")
        trusted = citation.get("trusted_for_claim_satisfaction", False)
        rows.append(
            f"- `{label}` ({section}) "
            f"[relevance={relevance}; supports_claims={supports_text}; "
            f"trusted_for_claim_satisfaction={trusted}]: {excerpt}"
        )
    return rows


def _render_claims(claims: list[dict]) -> list[str]:
    if not claims:
        return ["| No assurance claims available. | missing | yes | none | none |"]
    rows: list[str] = []
    for claim in claims:
        evidence = claim.get("evidence", [])
        first = evidence[0] if evidence else {}
        source = first.get("source", "none")
        if source == "retrieved_citations":
            source = (
                f"retrieved_citations"
                f"(relevance={first.get('relevance', 'low')}; "
                f"trusted={first.get('trusted_for_claim_satisfaction', False)})"
            )
        excerpt = str(first.get("text", "none")).replace("|", "\\|")
        status = str(claim.get("status", ""))
        if status == "required_change_only":
            status = "**required_change_only**"
        rows.append(
            "| "
            + " | ".join([
                str(claim.get("claim_id", "")),
                status,
                "yes" if claim.get("required_for_approval") else "no",
                source,
                excerpt[:260],
            ])
            + " |"
        )
    return rows


def _render_decision_trace(trace: dict) -> list[str]:
    if not trace:
        return ["- Decision trace unavailable."]
    return [
        f"- Final rule: `{trace.get('final_rule', '')}`",
        f"- Contradicted critical claims: {_list_text(trace.get('contradicted_critical_claims', []))}",
        f"- Missing critical claims: {_list_text(trace.get('missing_critical_claims', []))}",
        f"- Remediation-only critical claims: {_list_text(trace.get('remediation_only_critical_claims', []))}",
        f"- Untrusted citations ignored: {_list_text(trace.get('untrusted_citations_ignored', []))}",
        f"- Approval threshold: {trace.get('approval_threshold', '')}",
        f"- Risk threshold: {trace.get('risk_threshold', '')}",
    ]


def _render_llm_review(review: dict) -> list[str]:
    if not review:
        return ["- LLM review unavailable."]
    rows = [
        f"- Status: `{review.get('review_status', '')}`",
        f"- Model: `{review.get('model', '')}`",
        f"- Must not override: `{review.get('must_not_override', True)}`",
        f"- Summary: {review.get('reviewer_summary', '')}",
    ]
    findings = review.get("reviewer_findings", [])
    if findings:
        rows.extend(["", "| Type | Severity | Claim | Finding |", "|---|---|---|---|"])
        for finding in findings:
            rows.append(
                "| "
                + " | ".join([
                    str(finding.get("type", "")),
                    str(finding.get("severity", "")),
                    str(finding.get("claim_id", "")),
                    str(finding.get("text", "")).replace("|", "\\|")[:300],
                ])
                + " |"
            )
    return rows


def _render_external_judge(judge: dict, payload: dict) -> list[str]:
    if not judge:
        return ["- External judge unavailable."]
    rows = [
        f"- Provider: `{judge.get('provider', '')}`",
        f"- Model: `{judge.get('model', '')}`",
        f"- Status: `{judge.get('judge_status', '')}`",
        f"- Reason: `{judge.get('reason', '')}`",
        f"- Judge verdict recommendation: `{judge.get('judge_verdict_recommendation', 'unknown')}`",
        f"- Judge risk recommendation: `{judge.get('judge_risk_recommendation', 'unknown')}`",
        f"- Non-override verified: `{judge.get('external_judge_non_override_verified', False)}`",
    ]
    if judge.get("judge_verdict_recommendation") not in {None, "unknown", payload.get("verdict")}:
        rows.append(
            f"- Disagreement: judge recommended `{judge.get('judge_verdict_recommendation')}` "
            f"but deterministic verdict is `{payload.get('verdict')}`."
        )
    findings = judge.get("judge_findings", [])
    if findings:
        rows.extend(["", "| Type | Severity | Claim | Finding |", "|---|---|---|---|"])
        for finding in findings:
            rows.append(
                "| "
                + " | ".join([
                    str(finding.get("type", "")),
                    str(finding.get("severity", "")),
                    str(finding.get("claim_id", "")),
                    str(finding.get("text", "")).replace("|", "\\|")[:300],
                ])
                + " |"
            )
    else:
        rows.append("- Findings: none.")
    return rows


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None."]


def _humanize(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _list_text(items: list) -> str:
    return ", ".join(str(item) for item in items) if items else "none"

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assured_rag.api_keys import get_api_key


PRINCIPAL_PAPERS = [
    "M06_ExternalReadings-2024-open_autonomy_safety_case_2024-pdf",
    "M05_DataSetShift-Sinha-2022-SystemLevelOODRobotics.pdf",
    "M12_RuntimeAssurance-Desai-2019-SOTER.pdf",
    "M06_ExternalReadings-2023-waymo_safety_case_2023-pdf",
    "M01-02_Introduction-HernandezOrallo-2019-SurveyingSafetyRelevantAI-pdf",
]

REQUIRED_SECTIONS = [
    "## Not supported by this paper",
    "## Scope conditions",
    "## Compiler use",
    "## Weak-reader traps",
    "## Backend-specific card policy",
]

PAPER_HINTS = {
    "M06_ExternalReadings-2024-open_autonomy_safety_case_2024-pdf": {
        "week": "M06",
        "why": "Dominates non-Mistral metric/auditability attribution and safety-case evidence-vs-claim confusion.",
        "tags": ["safety_case", "evidence_vs_claim", "auditability", "unsupported_claim_metric_suspect"],
    },
    "M05_DataSetShift-Sinha-2022-SystemLevelOODRobotics.pdf": {
        "week": "M05",
        "why": "Dominates GPT-3.5 hard-stop/RAG-overload attribution through system-level OOD and transfer limits.",
        "tags": ["OOD", "testing_evaluation", "rag_overload_risk", "scope_conditions"],
    },
    "M12_RuntimeAssurance-Desai-2019-SOTER.pdf": {
        "week": "M12",
        "why": "Primary runtime-assurance hard-stop reference for Mistral verdict propagation failures.",
        "tags": ["runtime_assurance", "hard_stop_softening", "fallback_controller", "safety_envelope"],
    },
    "M06_ExternalReadings-2023-waymo_safety_case_2023-pdf": {
        "week": "M06",
        "why": "Major Mistral hard-stop/safety-case reference for risk acceptance and multi-method evidence.",
        "tags": ["safety_case", "verification_validation", "evidence_vs_claim", "risk_acceptance"],
    },
    "M01-02_Introduction-HernandezOrallo-2019-SurveyingSafetyRelevantAI-pdf": {
        "week": "M01-02",
        "why": "Safety-relevant AI framing used by Grok/auditability attribution and broad compiler routing.",
        "tags": ["safety_relevant_ai", "auditability", "scope_conditions", "testing_evaluation"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair the five principal paper cards with gated multi-LLM review.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-api-call", action="store_true")
    parser.add_argument("--papers", default="principal5")
    parser.add_argument("--output-plan", default="reports/principal_paper_repair_plan.md")
    parser.add_argument("--output-dir", default="reports/principal_repaired_cards")
    parser.add_argument("--audit-json", default="reports/principal_paper_repair_audit.json")
    parser.add_argument("--audit-csv", default="reports/principal_paper_repair_audit.csv")
    parser.add_argument("--crosswalk", default="reports/principal_paper_crosswalk.md")
    parser.add_argument("--quality-csv", default="reports/principal_paper_quality_scores.csv")
    parser.add_argument("--resume", action="store_true", help="Preserve existing repaired cards and continue missing cards.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing repaired cards.")
    parser.add_argument("--provider-timeout", type=float, default=90.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    papers = principal_papers(args.papers)
    inventories = [locate_paper(pid) for pid in papers]
    baseline = [baseline_audit(item) for item in inventories]
    provider_status = provider_availability(allow_api_call=args.allow_api_call, dry_run=args.dry_run)

    plan = render_plan(inventories, baseline, provider_status, dry_run=args.dry_run)
    plan_path = Path(args.output_plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan, encoding="utf-8")
    print(f"Wrote {plan_path}")

    if args.dry_run:
        return 0
    if not args.allow_api_call and os.environ.get("ASSURED_RAG_ALLOW_API_CALLS") != "1":
        print("API calls not allowed; wrote plan only.")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired = []
    for inventory, audit in zip(inventories, baseline):
        card_path = output_dir / f"{inventory['canonical_paper_id']}.repaired.md"
        if card_path.exists() and args.resume and not args.force:
            card = card_path.read_text(encoding="utf-8", errors="ignore")
            reviews = {"resume": {"status": "skipped", "reason": "existing_repaired_card_preserved"}}
            quality = quality_gate(card)
        else:
            reviews = run_multi_provider_reviews(inventory, audit, provider_status, timeout=args.provider_timeout)
            card = build_repaired_card(inventory, audit, reviews)
            quality = quality_gate(card)
            card_path.write_text(card, encoding="utf-8")
        repaired.append({
            "canonical_paper_id": inventory["canonical_paper_id"],
            "card_path": str(card_path),
            "quality": quality,
            "reviews": summarize_reviews(reviews),
            "baseline_audit": audit,
            "source_paths": inventory["source_paths"],
            "card_variants": inventory["card_variants"],
        })

    index_path = output_dir.parent / "principal_repaired_cards_index.md"
    index_path.write_text(render_index(repaired), encoding="utf-8")
    Path(args.crosswalk).write_text(render_crosswalk(repaired), encoding="utf-8")
    audit_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "principal_papers": papers,
        "provider_status": provider_status,
        "repaired_cards": repaired,
    }
    Path(args.audit_json).write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_audit_csv(repaired, Path(args.audit_csv))
    write_quality_csv(repaired, Path(args.quality_csv))
    print(f"Wrote {index_path}")
    print(f"Wrote {args.crosswalk}")
    print(f"Wrote {args.audit_json}")
    print(f"Wrote {args.audit_csv}")
    print(f"Wrote {args.quality_csv}")
    return 0


def principal_papers(selection: str) -> list[str]:
    if selection != "principal5":
        requested = [item.strip() for item in selection.split(",") if item.strip()]
        unknown = [item for item in requested if item not in PRINCIPAL_PAPERS]
        if unknown:
            raise SystemExit(f"Unknown principal paper id(s): {', '.join(unknown)}")
        return requested
    return list(PRINCIPAL_PAPERS)


def locate_paper(canonical_paper_id: str) -> dict[str, Any]:
    root = Path(".")
    tokens = token_variants(canonical_paper_id)
    source_paths = []
    card_variants = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(token.lower() in name for token in tokens):
            rel = str(path)
            if path.suffix.lower() == ".pdf":
                source_paths.append(rel)
            elif path.suffix.lower() in {".md", ".txt"} and ("card" in rel.lower() or "sweep" in rel.lower() or "semantic" in rel.lower()):
                card_variants.append(rel)
    card_variants = sorted(set(card_variants))
    source_paths = sorted(set(source_paths))
    text_sample = build_text_sample(source_paths, card_variants)
    return {
        "canonical_paper_id": canonical_paper_id,
        "hint": PAPER_HINTS.get(canonical_paper_id, {}),
        "source_paths": source_paths,
        "card_variants": card_variants,
        "text_sample": text_sample,
    }


def token_variants(canonical_paper_id: str) -> list[str]:
    base = canonical_paper_id.replace("-pdf", "").replace(".pdf", "")
    pieces = [base, base.replace("_", "-"), base.replace("-", "_")]
    if "open_autonomy_safety_case" in base:
        pieces.extend(["open_autonomy_safety_case", "open-autonomy-safety-case"])
    if "waymo_safety_case" in base:
        pieces.extend(["waymo_safety_case", "waymo-safety-case"])
    if "Sinha-2022" in base:
        pieces.append("Sinha-2022-SystemLevelOODRobotics")
    if "Desai-2019" in base:
        pieces.append("Desai-2019-SOTER")
    if "HernandezOrallo" in base:
        pieces.append("HernandezOrallo-2019-SurveyingSafetyRelevantAI")
    return pieces


def build_text_sample(source_paths: list[str], card_variants: list[str], max_chars: int = 24000) -> str:
    chunks = []
    for path in source_paths[:2]:
        text = extract_pdf_text(Path(path), max_chars=max_chars // 2)
        if text:
            chunks.append(f"PDF SOURCE {path}\n{text}")
    for path in card_variants[:5]:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")[:4000]
            chunks.append(f"CARD VARIANT {path}\n{text}")
        except Exception:
            continue
    return "\n\n".join(chunks)[:max_chars]


def extract_pdf_text(path: Path, max_chars: int) -> str:
    try:
        import fitz  # type: ignore
        doc = fitz.open(path)
        pages = []
        for index, page in enumerate(doc):
            if index >= 12:
                break
            pages.append(page.get_text())
            if sum(len(item) for item in pages) >= max_chars:
                break
        return "\n".join(pages)[:max_chars]
    except Exception:
        return ""


def baseline_audit(inventory: dict[str, Any]) -> dict[str, Any]:
    joined = "\n".join(read_file(path, 3000) for path in inventory["card_variants"][:4])
    lower = joined.lower()
    scores = {
        "faithfulness_to_source": score_by_terms(lower, ["source", "paper", "claim", "evidence"]),
        "definition_preservation": score_by_terms(lower, ["definition", "term", "concept"]),
        "scope_conditions": score_by_terms(lower, ["scope", "condition", "limitation", "does not"]),
        "evidence_vs_claim_separation": score_by_terms(lower, ["evidence", "claim", "unsupported", "do not claim"]),
        "hard_stop_relevance": score_by_terms(lower, ["reject", "unacceptable", "hard-stop", "fallback", "switch"]),
        "runtime_assurance_relevance": score_by_terms(lower, ["runtime assurance", "simplex", "fallback", "monitor"]),
        "testing_evaluation_relevance": score_by_terms(lower, ["test", "evaluation", "validation", "benchmark"]),
        "citation_granularity": min(5, lower.count("p.") + lower.count("page") + lower.count("pp.")),
        "risk_of_llm_distortion": 5 - score_by_terms(lower, ["not supported", "do not claim", "limitation", "scope"]),
    }
    flags = []
    checks = {
        "missing definitions": "definition" not in lower,
        "missing scope conditions": "scope" not in lower and "limitation" not in lower,
        "vague safety-case language": "safety" in lower and "evidence" not in lower,
        "unsupported hard-stop claims": "reject" in lower and "evidence" not in lower,
        "overgeneralization": "always" in lower or "guarantee" in lower,
        "collapse of testing evidence into assurance approval": "approve" in lower and "test" in lower and "condition" not in lower,
        "collapse of human oversight into containment": "human" in lower and "containment" not in lower,
        "missing not supported section": "not supported" not in lower and "do not claim" not in lower,
        "missing connection to current blindspots": not any(term in lower for term in ["hard_stop", "runtime", "auditability", "unsupported"]),
    }
    flags.extend([label for label, present in checks.items() if present])
    return {"scores": scores, "flags": flags, "variant_count": len(inventory["card_variants"])}


def score_by_terms(text: str, terms: list[str]) -> int:
    return min(5, sum(1 for term in terms if term in text))


def read_file(path: str, max_chars: int) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def provider_availability(*, allow_api_call: bool, dry_run: bool) -> dict[str, dict[str, Any]]:
    return {
        "openai_primary": provider_info("openai", os.environ.get("ASSURED_RAG_REPAIR_OPENAI_MODEL") or "gpt-5.5", bool(get_api_key("OPENAI_API_KEY")), allow_api_call, dry_run),
        "openai_sanity": provider_info("openai", os.environ.get("ASSURED_RAG_REPAIR_OPENAI_SANITY_MODEL") or "gpt-4.1-nano", bool(get_api_key("OPENAI_API_KEY")), allow_api_call, dry_run),
        "mistral_weak_reader": provider_info("mistral", os.environ.get("ASSURED_RAG_MISTRAL_MODEL") or "mistral-large-latest", bool(get_api_key("MISTRAL_API_KEY")), allow_api_call, dry_run),
        "grok_critique": provider_info("grok", os.environ.get("ASSURED_RAG_GROK_STABILITY_MODEL") or "grok-4.3", bool(get_api_key("XAI_API_KEY")), allow_api_call, dry_run),
    }


def provider_info(provider: str, model: str, has_key: bool, allow_api_call: bool, dry_run: bool) -> dict[str, Any]:
    status = "available" if has_key and allow_api_call and not dry_run else "skipped"
    reason = "" if status == "available" else ("dry_run" if dry_run else "api_call_not_allowed" if not allow_api_call else "missing_api_key")
    return {"provider": provider, "model": model, "has_key": has_key, "status": status, "reason": reason}


def run_multi_provider_reviews(inventory: dict[str, Any], audit: dict[str, Any], provider_status: dict[str, dict[str, Any]], *, timeout: float = 90.0) -> dict[str, Any]:
    roles = {
        "openai_primary": "Primary extractor, assurance-case analyst, runtime-assurance analyst, and cross-examiner.",
        "openai_sanity": "Structured extraction sanity checker.",
        "mistral_weak_reader": "Weak-reader adversary: intentionally identify likely misreadings and failure modes.",
        "grok_critique": "Independent critique focused on evidence IDs, auditability, and unsupported claims.",
    }
    reviews = {}
    for name, info in provider_status.items():
        if info["status"] != "available":
            reviews[name] = {"status": "skipped", "reason": info["reason"], "model": info["model"], "provider": info["provider"]}
            continue
        prompt = build_repair_prompt(inventory, audit, roles[name])
        reviews[name] = call_provider(info["provider"], info["model"], prompt, timeout=timeout)
    return reviews


def build_repair_prompt(inventory: dict[str, Any], audit: dict[str, Any], role: str) -> str:
    return f"""You are repairing a paper card for an Assured Autonomy RAG compiler.

Role: {role}

Safety boundary:
- Do not provide operational cyber, targeting, attack, maneuver, exploit, or live-control guidance.
- Focus only on assurance concepts, evidence requirements, scope conditions, auditability, runtime assurance, and safe compiler use.
- Separate direct claims from inference. Do not invent facts.

Paper ID: {inventory['canonical_paper_id']}
Why principal: {inventory['hint'].get('why', '')}
Current audit flags: {', '.join(audit.get('flags', []))}

Source/card excerpts:
{inventory.get('text_sample', '')}

Return compact markdown notes with:
1. direct claims
2. definitions
3. scope conditions
4. evidence-vs-claim cautions
5. hard-stop/runtime-assurance implications if justified
6. weak-reader traps
7. not-supported claims
"""


def call_provider(provider: str, model: str, prompt: str, *, timeout: float) -> dict[str, Any]:
    if provider == "openai":
        return call_openai(model, prompt, timeout=timeout)
    if provider == "mistral":
        return call_chat_completion("mistral", model, "https://api.mistral.ai/v1/chat/completions", get_api_key("MISTRAL_API_KEY"), prompt, timeout=timeout)
    if provider == "grok":
        return call_chat_completion("grok", model, "https://api.x.ai/v1/chat/completions", get_api_key("XAI_API_KEY"), prompt, timeout=timeout)
    return {"status": "skipped", "reason": "unsupported_provider", "model": model, "provider": provider}


def call_openai(model: str, prompt: str, *, timeout: float) -> dict[str, Any]:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"), timeout=timeout)
        response = client.responses.create(
            model=model,
            max_output_tokens=1800,
            input=[
                {"role": "system", "content": "You produce conservative paper-card repair notes for assurance RAG. No operational guidance."},
                {"role": "user", "content": prompt},
            ],
        )
        return {"status": "ok", "provider": "openai", "model": model, "text": (getattr(response, "output_text", None) or "")[:20000]}
    except Exception as exc:
        return {"status": "error", "provider": "openai", "model": model, "reason": classify_error(exc), "detail": str(exc)[:500]}


def call_chat_completion(provider: str, model: str, url: str, api_key: str, prompt: str, *, timeout: float) -> dict[str, Any]:
    if not api_key:
        return {"status": "skipped", "provider": provider, "model": model, "reason": "missing_api_key"}
    try:
        import requests  # type: ignore
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.1,
                "max_tokens": 1800,
                "messages": [
                    {"role": "system", "content": "You produce conservative paper-card repair notes for assurance RAG. No operational guidance."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return {"status": "ok", "provider": provider, "model": model, "text": response.json()["choices"][0]["message"]["content"][:20000]}
    except Exception as exc:
        return {"status": "error", "provider": provider, "model": model, "reason": classify_error(exc), "detail": str(exc)[:500]}


def classify_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if code in {401, 403}:
        return "auth_or_key_format_failure"
    if code in {400, 404}:
        return "invalid_model_or_request"
    if code == 429:
        return "rate_limit"
    return "request_error"


def build_repaired_card(inventory: dict[str, Any], audit: dict[str, Any], reviews: dict[str, Any]) -> str:
    pid = inventory["canonical_paper_id"]
    hint = inventory.get("hint", {})
    review_text = "\n\n".join(
        f"### {name} ({review.get('status')})\n{review.get('text', review.get('reason', ''))}"
        for name, review in reviews.items()
    )
    flags = audit.get("flags", [])
    tags = sorted(set(hint.get("tags", []) + [
        "hard_stop_softening",
        "runtime_assurance",
        "testing_evaluation",
        "evidence_vs_claim",
        "safety_case",
        "OOD",
        "human_oversight_not_containment",
        "auditability",
        "rag_overload_risk",
    ]))
    return f"""# Principal Paper Card: {pid}

## Canonical ID
{pid}

## Source
- Principal source paths: {', '.join(inventory.get('source_paths', [])[:5]) or 'source PDF not located'}
- Week/module: {hint.get('week', 'unknown')}
- Existing card variants inspected: {len(inventory.get('card_variants', []))}
- Original cards preserved: yes. This repaired card is written under `reports/principal_repaired_cards/`.

## Why this paper is principal
{hint.get('why', 'Selected by paper-level confusion attribution.')}

Implicated blindspots: hard_stop_softening, rag_overload_or_confusion, auditability_failure, unsupported_claim_metric_suspect, evidence-vs-claim confusion, human/operator approval treated as containment, runtime-assurance hard-stop propagation.

## Direct claims from the paper
Use only claims directly supported by the source/card excerpts and provider notes. Treat the following as conservative direct-claim candidates:
- The paper contributes assurance-relevant structure, not blanket deployment approval.
- Evidence must be tied to a claim, operational context, and scope condition.
- Testing, monitoring, or safety-case language does not by itself prove acceptability.

Provider notes:
{review_text or 'No provider notes available; dry-run or all providers skipped.'}

## Key definitions
- Assurance claim: a claim requiring explicit evidence and scope.
- Scope condition: the operational context under which evidence applies.
- Hard stop: a decision rule that blocks approval when risk remains high/unacceptable.
- Runtime assurance: monitor/controller/fallback logic that constrains behavior during operation, when supported by the paper.

## Scope conditions
- Applies when the compiler is reasoning about assurance evidence, test validity, runtime monitoring, or claim/evidence traceability.
- Does not apply when the proposal lacks bounded synthetic/offline scope.
- Requires evidence of boundary, non-interference, audit trail, and claim-specific support.
- Human review is oversight evidence only; it is not containment by itself.

## Evaluation / assurance relevance
- Testing/evaluation relevance: supports asking whether evidence is adequate for the claim and context.
- V&V relevance: supports separating validation methodology from approval.
- Runtime-assurance relevance: supports hard-stop or fallback reasoning only when the paper actually discusses monitor/controller/safety-envelope concepts.
- Assurance-case relevance: supports claim/evidence discipline and explicit “not supported” boundaries.
- OOD/system-level relevance, if applicable: supports warnings about transfer from one context to another.

## Hard-stop implications
- Conditions that should force reject/unacceptable: high or unacceptable risk, live access, operational outputs, missing containment, or evidence that does not support the assurance claim.
- Conditions that justify revise: bounded proposal with missing scope, auditability, or evidence details.
- Conditions that might permit approve: all approval-critical claims are satisfied by proposal text or trusted, relevant retrieved evidence.
- Required evidence: explicit system boundary, synthetic/offline environment, no live cyber/telemetry/actuation, no real targeting data, no operational outputs, logging/replay, non-interference.

## Human oversight / operator approval
Human review can support oversight and review quality. It is not containment. Operator approval does not repair live actuation, operational recommendations, missing sandboxing, or missing non-interference evidence.

## Evidence vs claim discipline
- What the paper supports: structured reasoning about claims, scope, evidence, testing, monitoring, and limits.
- What it does not support: approving a proposal merely because it uses “safety case,” “red team,” “defensive,” “human supervised,” or “stress test” language.
- Common misuse patterns: treating evidence existence as evidence sufficiency; treating mitigation plans as proof; treating human review as containment.

## RAG routing tags
{', '.join(tags)}

## Not supported by this paper
- Do not claim the paper proves a system is safe in arbitrary deployment contexts.
- Do not claim test evidence automatically justifies approval.
- Do not claim human/operator approval is equivalent to containment.
- Do not claim a runtime-assurance concept applies unless monitor, fallback, switching, or safety-envelope conditions are actually present.
- Do not use this paper to soften reject/unacceptable into revise when high/unacceptable risk remains.

## Weak-reader traps
- A weak reader may see assurance vocabulary and infer approval.
- A weak reader may recognize unacceptable risk but still recommend revise.
- A weak reader may conflate retrieved evidence with claim satisfaction.
- A weak reader may treat human review as the safety mechanism rather than oversight.
- A weak reader may overload on long multi-card packets and ignore the hard-stop rule.

## Backend-specific card policy
### Mistral policy
- Hard-stop wording must be explicit.
- Include consistency rule: high/unacceptable risk cannot map to approve/revise.
- Prefer this card only when the hard-stop trigger is present.

### GPT-3.5 policy
- Use a smaller evidence packet.
- Include at most 2-3 snippets.
- Put final rule before evidence.
- Avoid long multi-card packets that cause RAG overload.

### GPT-4.1-nano policy
- Preserve evidence IDs.
- Separate caveats from unsupported claims.
- Use for auditable RAG-grounded behavior.

### Grok policy
- Force `evidence_ids_used`.
- Flag missing evidence IDs as auditability failure.

## Compiler use
- Retrieval triggers: {', '.join(tags)}
- Include in evidence packet when the proposal asks about assurance claims, runtime fallback, test validity, operator approval, or evidence sufficiency.
- Avoid including long versions of this card to weaker models; use a compact snippet plus hard-stop consistency rule.
- Never allow this card to override deterministic safety flags.

## Baseline audit issues repaired
{', '.join(flags) if flags else 'No major baseline issues detected.'}
"""


def quality_gate(card: str) -> dict[str, Any]:
    missing = [section for section in REQUIRED_SECTIONS if section not in card]
    risky = []
    if unsupported_guarantee_sentence(card):
        risky.append("unsupported guarantee wording")
    return {"accepted": not missing and not risky, "missing_sections": missing, "risk_flags": risky}


def unsupported_guarantee_sentence(card: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+|\n", card.lower()):
        if "guarantees safety" not in sentence:
            continue
        if any(guard in sentence for guard in ["do not", "does not", "not claim", "unsupported", "weak reader", "trap", "proof that", "assumption that", "without", " if "]):
            continue
        return True
    return False


def summarize_reviews(reviews: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "provider": review.get("provider"),
            "model": review.get("model"),
            "status": review.get("status"),
            "reason": review.get("reason", ""),
        }
        for name, review in reviews.items()
    }


def render_plan(inventories: list[dict[str, Any]], audits: list[dict[str, Any]], provider_status: dict[str, Any], *, dry_run: bool) -> str:
    lines = [
        "# Principal Paper Repair Plan",
        "",
        f"Mode: {'dry run' if dry_run else 'repair run'}",
        "",
        "## Provider Status",
        "",
        "| Role | Provider | Model | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for role, info in provider_status.items():
        lines.append(f"| {role} | {info['provider']} | {info['model']} | {info['status']} | {info.get('reason','')} |")
    lines.extend(["", "## Papers", "", "| Canonical paper | Sources | Card variants | Main audit flags |", "|---|---:|---:|---|"])
    for inventory, audit in zip(inventories, audits):
        lines.append(
            f"| {inventory['canonical_paper_id']} | {len(inventory['source_paths'])} | {len(inventory['card_variants'])} | "
            f"{', '.join(audit['flags'][:6])} |"
        )
    lines.extend([
        "",
        "## Planned Repair Protocol",
        "",
        "For each principal paper: locate source/card variants, audit existing card quality, run gated provider roles, write repaired card under reports/principal_repaired_cards, and preserve all original cards.",
    ])
    return "\n".join(lines).strip() + "\n"


def render_index(repaired: list[dict[str, Any]]) -> str:
    lines = ["# Principal Repaired Cards Index", "", "| Paper | Repaired card | Accepted | Providers |", "|---|---|---:|---|"]
    for item in repaired:
        providers = ", ".join(f"{k}:{v['status']}" for k, v in item["reviews"].items())
        lines.append(f"| {item['canonical_paper_id']} | `{item['card_path']}` | {item['quality']['accepted']} | {providers} |")
    return "\n".join(lines).strip() + "\n"


def render_crosswalk(repaired: list[dict[str, Any]]) -> str:
    clusters = {
        "Safety-case evidence cluster": ["open_autonomy", "waymo", "Seshia", "M06"],
        "Runtime-assurance cluster": ["SOTER", "RuntimeAssurance", "M12"],
        "Evaluation/OOD cluster": ["Sinha", "OOD", "M05"],
        "Safety framing cluster": ["HernandezOrallo", "M01-02"],
    }
    lines = ["# Principal Paper Crosswalk", ""]
    for cluster, needles in clusters.items():
        lines.extend([f"## {cluster}", ""])
        matched = [
            item for item in repaired
            if any(needle.lower() in item["canonical_paper_id"].lower() for needle in needles)
        ]
        if not matched:
            lines.append("No principal repaired card in this cluster.")
        for item in matched:
            lines.append(f"- `{item['canonical_paper_id']}` repairs blindspots: {', '.join(PAPER_HINTS.get(item['canonical_paper_id'], {}).get('tags', []))}")
        lines.append("")
    lines.extend([
        "## Blindspot-Aware RAG Architecture",
        "",
        "Use safety-case cards for evidence-vs-claim discipline, SOTER for runtime hard-stop propagation, Sinha for system-level OOD/scope transfer, and Hernandez-Orallo for safety-relevant AI framing. For weaker backends, route shorter snippets with explicit consistency rules before evidence.",
    ])
    return "\n".join(lines).strip() + "\n"


def write_audit_csv(repaired: list[dict[str, Any]], path: Path) -> None:
    fields = ["canonical_paper_id", "card_path", "accepted", "missing_sections", "risk_flags", "provider_statuses"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in repaired:
            writer.writerow({
                "canonical_paper_id": item["canonical_paper_id"],
                "card_path": item["card_path"],
                "accepted": item["quality"]["accepted"],
                "missing_sections": ";".join(item["quality"]["missing_sections"]),
                "risk_flags": ";".join(item["quality"]["risk_flags"]),
                "provider_statuses": json.dumps(item["reviews"]),
            })


def write_quality_csv(repaired: list[dict[str, Any]], path: Path) -> None:
    fields = ["canonical_paper_id", *next(iter(repaired), {"baseline_audit": {"scores": {}}})["baseline_audit"]["scores"].keys()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in repaired:
            row = {"canonical_paper_id": item["canonical_paper_id"], **item["baseline_audit"]["scores"]}
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())

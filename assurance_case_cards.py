"""Compile council RAG runs into reusable assurance-case cards.

The council runs are rich prose. This module turns them into structured memory:
what the artifact is, what assurance role it can play, where its cyber-physical
escalation point is, what evidence is missing, and what a student should not
claim. These cards are intended to feed the concept graph before broad reading
list sweeps or project synthesis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .bridge_graph_rag import _json_from_text
from .llm import get_llm


DEFAULT_OUT_DIR = Path("assurance_case_runs")
REQUIRED_FIELDS = [
    "artifact_name",
    "artifact_kind",
    "student_audience",
    "assurance_role",
    "recommended_project_role",
    "hidden_witnesses",
    "claim_layers",
    "cyberphysical_escalation_points",
    "physical_world_assumptions",
    "missing_evidence",
    "needed_evidence",
    "runtime_monitoring_hooks",
    "tests_or_metrics",
    "risk_portfolio",
    "course_grounding",
    "code_annotation_tags",
    "trust_transfer_questions",
    "overclaim_warnings",
    "next_actions",
    "confidence",
]


def _safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")
    return text[:90] or "artifact"


def _clean(text: Any, limit: int = 1200) -> str:
    if isinstance(text, (dict, list)):
        text = json.dumps(text, ensure_ascii=False)
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit].rstrip()


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _infer_artifact_name(query: str, run_path: Path) -> str:
    q = query.lower()
    aliases = [
        ("AlpaSim", ["alpasim"]),
        ("BadgerGuard", ["badgerguard"]),
        ("BarrierNet", ["barriernet", "barrier net"]),
        ("GTSRB YOLO", ["gtsrb", "yolo"]),
        ("Hybrid CBF", ["hybrid cbf"]),
        ("MapDR", ["mapdr"]),
        ("NeuroNCAP", ["neuroncap", "neuro ncap"]),
        ("Ogma", ["ogma"]),
        ("REAP", ["reap"]),
        ("ROBOSAC", ["robosac"]),
        ("ROSMonitoring", ["rosmonitoring", "ros monitoring"]),
        ("safe_control", ["safe_control", "safe control"]),
        ("ScenarioRunner", ["scenario runner", "scenariorunner"]),
        ("Scenic", ["scenic"]),
        ("SP_RL", ["sp_rl", "sp rl"]),
        ("Trusted-RL-for-Autonomous-Driving", ["trusted-rl", "trusted rl"]),
        ("Zhang ZKP Motion Planning", ["zhang", "zero-knowledge", "zkp"]),
    ]
    for label, needles in aliases:
        if any(needle in q for needle in needles):
            return label
    match = re.search(r"(?:repo|repository|project|codebase)\s*[:=]\s*([^\n]+)", query, re.I)
    if match:
        return _clean(match.group(1), 100)
    return run_path.parent.name


def _normalise_card(data: dict[str, Any], *, run_path: Path, source_payload: dict[str, Any]) -> dict[str, Any]:
    query = str(source_payload.get("query") or "")
    card: dict[str, Any] = {
        "assurance_case_version": "1.0",
        "source_run": str(run_path),
        "source_models": source_payload.get("models", {}),
        "artifact_name": _infer_artifact_name(query, run_path),
        "artifact_kind": "unknown",
        "student_audience": "Hopkins cybersecurity-track student studying assured autonomy",
        "assurance_role": "",
        "recommended_project_role": "",
        "hidden_witnesses": [],
        "claim_layers": {
            "software_or_formal_claim": "",
            "encoded_model_claim": "",
            "cyberphysical_behavior_claim": "",
            "runtime_conformance_claim": "",
            "trust_transfer_claim": "",
        },
        "cyberphysical_escalation_points": [],
        "physical_world_assumptions": [],
        "missing_evidence": [],
        "needed_evidence": [],
        "runtime_monitoring_hooks": [],
        "tests_or_metrics": [],
        "risk_portfolio": {
            "risky_assets": [],
            "risk_factors": [],
            "exposure_vector": {},
            "controls_or_hedges": [],
            "evidence_assets": [],
            "residual_risk": [],
            "portfolio_role": "",
        },
        "course_grounding": [],
        "code_annotation_tags": [],
        "trust_transfer_questions": [],
        "overclaim_warnings": [],
        "next_actions": [],
        "confidence": "medium",
    }
    card.update({k: v for k, v in data.items() if v not in (None, "")})
    for field in REQUIRED_FIELDS:
        if field not in card:
            card[field] = [] if field not in {"artifact_name", "artifact_kind", "student_audience", "assurance_role", "recommended_project_role", "confidence", "claim_layers"} else ""
    for field in [
        "hidden_witnesses",
        "cyberphysical_escalation_points",
        "physical_world_assumptions",
        "missing_evidence",
        "needed_evidence",
        "runtime_monitoring_hooks",
        "tests_or_metrics",
        "course_grounding",
        "code_annotation_tags",
        "trust_transfer_questions",
        "overclaim_warnings",
        "next_actions",
    ]:
        card[field] = _listify(card.get(field))[:18]
    if not isinstance(card.get("claim_layers"), dict):
        card["claim_layers"] = {"summary": _clean(card.get("claim_layers"))}
    if not isinstance(card.get("risk_portfolio"), dict):
        card["risk_portfolio"] = {
            "risky_assets": [],
            "risk_factors": [],
            "exposure_vector": {},
            "controls_or_hedges": [],
            "evidence_assets": [],
            "residual_risk": [_clean(card.get("risk_portfolio"))],
            "portfolio_role": "",
        }
    card["artifact_name"] = _clean(card["artifact_name"], 120)
    card["artifact_kind"] = _clean(card["artifact_kind"], 160)
    card["student_audience"] = _clean(card["student_audience"], 220)
    card["assurance_role"] = _clean(card["assurance_role"], 700)
    card["recommended_project_role"] = _clean(card["recommended_project_role"], 700)
    card["confidence"] = _clean(card["confidence"], 80) or "medium"
    return card


def _fallback_card(run_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    lift = payload.get("lift") or {}
    gate = payload.get("evidence_gate") or {}
    query = str(payload.get("query") or "")
    return _normalise_card(
        {
            "artifact_name": _infer_artifact_name(query, run_path),
            "artifact_kind": "RAG-analyzed artifact",
            "assurance_role": _clean(payload.get("final_answer") or payload.get("scout_answer"), 700),
            "recommended_project_role": "Use as a bounded assurance-case component after checking missing evidence and overclaim warnings.",
            "hidden_witnesses": lift.get("hidden_witnesses", []),
            "claim_layers": {
                "software_or_formal_claim": _clean(lift.get("proof_or_evidence_claims", []), 500),
                "encoded_model_claim": _clean(lift.get("behavioral_claims", []), 500),
                "cyberphysical_behavior_claim": _clean(lift.get("cyber_to_cps_escalation_points", []), 500),
                "runtime_conformance_claim": "Requires explicit runtime monitoring or fallback evidence.",
                "trust_transfer_claim": _clean(lift.get("trust_transfer_questions", []), 500),
            },
            "cyberphysical_escalation_points": lift.get("cyber_to_cps_escalation_points", []),
            "physical_world_assumptions": lift.get("physical_world_assumptions", []),
            "missing_evidence": lift.get("assurance_gaps", []),
            "needed_evidence": gate.get("suggested_expansion_terms", []),
            "runtime_monitoring_hooks": [],
            "tests_or_metrics": [],
            "risk_portfolio": {
                "risky_assets": ["autonomous system behavior", "safety claim", "student assurance argument"],
                "risk_factors": lift.get("assurance_gaps", []),
                "exposure_vector": {},
                "controls_or_hedges": gate.get("suggested_expansion_terms", []),
                "evidence_assets": gate.get("recommendation_summary", []),
                "residual_risk": lift.get("what_not_to_overclaim", []),
                "portfolio_role": "Fallback risk sketch from lift and evidence gate.",
            },
            "course_grounding": gate.get("recommendation_summary", []),
            "code_annotation_tags": [],
            "trust_transfer_questions": lift.get("trust_transfer_questions", []),
            "overclaim_warnings": lift.get("what_not_to_overclaim", []),
            "next_actions": [],
            "confidence": gate.get("status", "medium"),
        },
        run_path=run_path,
        source_payload=payload,
    )


def compile_assurance_card(
    run_path: Path,
    *,
    provider: str,
    model: str,
    max_tokens: int,
    no_llm: bool = False,
) -> dict[str, Any]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    if no_llm or provider == "stub":
        return _fallback_card(run_path, payload)
    llm = get_llm(provider, model=model)
    artifact = _infer_artifact_name(str(payload.get("query") or ""), run_path)
    compact = {
        "artifact_name_hint": artifact,
        "query": payload.get("query", "")[:6000],
        "models": payload.get("models", {}),
        "lift": payload.get("lift", {}),
        "evidence_gate": payload.get("evidence_gate", {}),
        "matched_bridge_cards": payload.get("matched_bridge_cards", [])[:4],
        "scout_answer": payload.get("scout_answer", "")[:9000],
        "critic_answer": payload.get("critic_answer", "")[:7000],
        "final_answer": payload.get("final_answer", "")[:12000],
    }
    system = (
        "You are the assurance-case compiler for a Hopkins Assured Autonomy RAG. "
        "Do not write an essay. Convert the council run into one reusable JSON card "
        "for cybersecurity-track students. Preserve claim layers and overclaim limits. "
        "Use only evidence present in the council run; mark gaps as missing evidence."
    )
    user = (
        "Council run payload:\n"
        f"{json.dumps(compact, ensure_ascii=False)}\n\n"
        "Return only JSON with exactly these top-level fields:\n"
        "{\n"
        '  "artifact_name": "...",\n'
        '  "artifact_kind": "repo | paper | scenario | tool | dataset | policy | other",\n'
        '  "student_audience": "Hopkins cybersecurity-track student ...",\n'
        '  "assurance_role": "...",\n'
        '  "recommended_project_role": "...",\n'
        '  "hidden_witnesses": ["..."],\n'
        '  "claim_layers": {\n'
        '    "software_or_formal_claim": "...",\n'
        '    "encoded_model_claim": "...",\n'
        '    "cyberphysical_behavior_claim": "...",\n'
        '    "runtime_conformance_claim": "...",\n'
        '    "trust_transfer_claim": "..."\n'
        "  },\n"
        '  "cyberphysical_escalation_points": ["..."],\n'
        '  "physical_world_assumptions": ["..."],\n'
        '  "missing_evidence": ["..."],\n'
        '  "needed_evidence": ["..."],\n'
        '  "runtime_monitoring_hooks": ["..."],\n'
        '  "tests_or_metrics": ["..."],\n'
        '  "risk_portfolio": {\n'
        '    "risky_assets": ["vehicle trajectory safety", "sensor integrity", "student assurance claim"],\n'
        '    "risk_factors": ["dataset shift", "monitor latency", "adversarial perception"],\n'
        '    "exposure_vector": {"dataset_shift": "high", "runtime_failure": "medium"},\n'
        '    "controls_or_hedges": ["runtime monitor", "scenario diversification", "safe controller"],\n'
        '    "evidence_assets": ["logs", "scenario results", "monitor verdicts"],\n'
        '    "residual_risk": ["simulation-to-real gap"],\n'
        '    "portfolio_role": "what role this artifact plays in reducing or measuring risk"\n'
        "  },\n"
        '  "course_grounding": [{"module": "M12 Runtime Assurance", "paper": "M12/example.pdf", "supports": "..."}],\n'
        '  "code_annotation_tags": [{"tag": "@runtime_monitor", "meaning": "...", "use_when": "..."}],\n'
        '  "trust_transfer_questions": ["..."],\n'
        '  "overclaim_warnings": ["..."],\n'
        '  "next_actions": ["..."],\n'
        '  "confidence": "low | medium | high"\n'
        "}\n"
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=max_tokens)
    data = _json_from_text(raw)
    if not data:
        data = _fallback_card(run_path, payload)
        data["compiler_warning"] = "LLM compiler returned unparseable JSON; used fallback."
        data["raw_compiler_output"] = raw[:3000]
    return _normalise_card(data, run_path=run_path, source_payload=payload)


def render_card_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# Assurance Case Card: {card['artifact_name']}",
        "",
        f"- Kind: {card.get('artifact_kind', '')}",
        f"- Confidence: {card.get('confidence', '')}",
        f"- Source run: `{card.get('source_run', '')}`",
        "",
        "## Assurance Role",
        card.get("assurance_role", ""),
        "",
        "## Recommended Project Role",
        card.get("recommended_project_role", ""),
        "",
        "## Claim Layers",
    ]
    for key, value in (card.get("claim_layers") or {}).items():
        lines.append(f"- **{key}:** {_clean(value, 500)}")
    for title, field in [
        ("Hidden Witnesses", "hidden_witnesses"),
        ("Cyber-Physical Escalation Points", "cyberphysical_escalation_points"),
        ("Physical-World Assumptions", "physical_world_assumptions"),
        ("Missing Evidence", "missing_evidence"),
        ("Needed Evidence", "needed_evidence"),
        ("Runtime Monitoring Hooks", "runtime_monitoring_hooks"),
        ("Tests Or Metrics", "tests_or_metrics"),
        ("Trust-Transfer Questions", "trust_transfer_questions"),
        ("Overclaim Warnings", "overclaim_warnings"),
        ("Next Actions", "next_actions"),
    ]:
        lines.extend(["", f"## {title}"])
        for item in card.get(field, []) or []:
            lines.append(f"- {_clean(item, 500)}")
    lines.extend(["", "## Risk Portfolio"])
    risk = card.get("risk_portfolio") or {}
    if isinstance(risk, dict):
        if risk.get("portfolio_role"):
            lines.append(f"- **Role:** {_clean(risk.get('portfolio_role'), 700)}")
        for key in ["risky_assets", "risk_factors", "controls_or_hedges", "evidence_assets", "residual_risk"]:
            vals = risk.get(key) or []
            if vals:
                lines.append(f"- **{key}:**")
                for item in vals[:8]:
                    lines.append(f"  - {_clean(item, 500)}")
        if risk.get("exposure_vector"):
            lines.append("- **exposure_vector:**")
            for k, v in list(risk["exposure_vector"].items())[:10]:
                lines.append(f"  - {k}: {_clean(v, 120)}")
    lines.extend(["", "## Course Grounding"])
    for item in card.get("course_grounding", []) or []:
        if isinstance(item, dict):
            label = item.get("paper") or item.get("filename") or item.get("module") or item
            support = item.get("supports") or item.get("rationale") or item.get("module_name") or ""
            lines.append(f"- {_clean(label, 220)}: {_clean(support, 500)}")
        else:
            lines.append(f"- {_clean(item, 600)}")
    lines.extend(["", "## Code Annotation Tags"])
    for item in card.get("code_annotation_tags", []) or []:
        if isinstance(item, dict):
            lines.append(f"- `{item.get('tag', '')}`: {_clean(item.get('meaning') or item.get('use_when'), 500)}")
        else:
            lines.append(f"- {_clean(item, 500)}")
    lines.append("")
    return "\n".join(lines)


def _iter_run_jsons(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in paths:
        if root.is_file() and root.name.endswith(".json"):
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob("council_bridge_rag.json")))
    return sorted(dict.fromkeys(found))


def build_cards(args: argparse.Namespace) -> int:
    run_paths = _iter_run_jsons([Path(p) for p in args.run_root])
    if args.max_runs:
        run_paths = run_paths[: args.max_runs]
    out_dir = Path(args.out_dir) / f"assurance_cases_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    for idx, run_path in enumerate(run_paths, 1):
        print(f"[{idx}/{len(run_paths)}] compile {run_path}")
        card = compile_assurance_card(
            run_path,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            no_llm=args.no_llm,
        )
        cards.append(card)
        stem = f"{idx:02d}_{_safe_name(card['artifact_name'])}"
        (cards_dir / f"{stem}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        (cards_dir / f"{stem}.md").write_text(render_card_markdown(card), encoding="utf-8")
    index = {
        "created_at": dt.datetime.now().isoformat(),
        "provider": args.provider,
        "model": args.model,
        "source_roots": args.run_root,
        "cards": [
            {
                "artifact_name": c.get("artifact_name"),
                "artifact_kind": c.get("artifact_kind"),
                "confidence": c.get("confidence"),
                "assurance_role": c.get("assurance_role"),
                "source_run": c.get("source_run"),
            }
            for c in cards
        ],
    }
    (out_dir / "assurance_case_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Assurance Case Card Index", ""]
    for row in index["cards"]:
        lines.append(f"- **{row['artifact_name']}** ({row['artifact_kind']}, confidence={row['confidence']})")
        lines.append(f"  - {row['assurance_role'][:260]}")
    (out_dir / "assurance_case_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved assurance case directory: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Compile council RAG runs into assurance-case cards")
    build.add_argument("--run-root", action="append", required=True,
                       help="Council run directory, root directory, or council_bridge_rag.json. Repeatable.")
    build.add_argument("--provider", default="openai",
                       choices=["openai", "xai", "grok", "mistral", "anthropic", "ollama", "stub", "peft-local"])
    build.add_argument("--model", default="gpt-5.5")
    build.add_argument("--max-tokens", type=int, default=5000)
    build.add_argument("--max-runs", type=int)
    build.add_argument("--no-llm", action="store_true",
                       help="Build deterministic fallback cards from existing lift/gate data.")
    build.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    build.set_defaults(func=build_cards)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Bridge-Graph RAG for abstraction-preserving course retrieval.

This layer sits before the existing corpus recommender. It captures "hidden"
expert bridges that do not show up reliably through keyword/vector retrieval:
toy problems, assurance principles, threat models, and module-spanning motifs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus_recommender import recommend_corpus, render_markdown as render_recs
from .config import AA_MODULES
from .llm import get_llm


DEFAULT_BRIDGE_DIR = Path("bridge_cards")
DEFAULT_OUT_DIR = Path("bridge_runs")


@dataclass
class BridgeNode:
    id: str
    kind: str
    label: str
    description: str = ""
    modules: list[str] | None = None
    source: str = "curated"


@dataclass
class BridgeEdge:
    source: str
    relation: str
    target: str
    rationale: str = ""


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return text or "item"


def _json_from_text(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalise_lift(data: dict[str, Any], raw: str, *, attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    keys = {
        "literal_terms": [],
        "lifted_concepts": [],
        "hidden_witnesses": [],
        "behavioral_claims": [],
        "proof_or_evidence_claims": [],
        "cyber_to_cps_escalation_points": [],
        "physical_world_assumptions": [],
        "assurance_gaps": [],
        "trust_transfer_questions": [],
        "verifier_questions": [],
        "toy_model_interpretations": [],
        "possible_le_cps_instantiations": [],
        "threat_models": [],
        "module_hypotheses": [],
        "retrieval_expansions": [],
        "what_not_to_overclaim": [],
        "bridge_summary": "",
    }
    out = {**keys, **data}
    for key in [
        "literal_terms",
        "lifted_concepts",
        "hidden_witnesses",
        "behavioral_claims",
        "proof_or_evidence_claims",
        "cyber_to_cps_escalation_points",
        "physical_world_assumptions",
        "assurance_gaps",
        "trust_transfer_questions",
        "verifier_questions",
        "toy_model_interpretations",
        "possible_le_cps_instantiations",
        "threat_models",
        "module_hypotheses",
        "retrieval_expansions",
        "what_not_to_overclaim",
    ]:
        out[key] = _listify(out.get(key))
    out["bridge_summary"] = str(out.get("bridge_summary") or "").strip()
    out["raw_model_output"] = raw
    if attempts:
        out["lift_attempts"] = attempts
    return out


def _heuristic_lift(query: str, *, raw: str = "", attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Deterministic backup for external anchors when the LLM returns no visible text."""
    q = query.lower()
    literal_terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)[:40]
    rules: list[tuple[list[str], str, str, str, str]] = [
        (["runtime", "monitor"], "runtime assurance as operational safety monitoring", "M12 Runtime Assurance", "runtime assurance monitor fallback safe controller", "Do not treat monitoring as a complete design-time proof."),
        (["fallback", "mitigation", "recovery"], "safe fallback and recovery under uncertainty", "M12 Runtime Assurance", "Simplex runtime assurance fallback recovery autonomous systems", "Do not claim fallback preserves mission optimality."),
        (["formal", "verification", "guarantee"], "formal guarantees for learning-enabled components", "M06 Verification and Validation", "formal verification neural network controller autonomous systems", "Do not claim formal verification scales to the full open world."),
        (["ltl", "temporal logic", "specification"], "temporal-logic constrained autonomy", "M06 Verification and Validation", "linear temporal logic controller synthesis autonomous robot safety", "Do not conflate satisfying an abstract formula with full real-world safety."),
        (["reinforcement", "rl", "safe rl"], "safe reinforcement learning with constrained objectives", "M13 Safe Reinforcement Learning", "safe reinforcement learning action projection safety constraints", "Do not claim learned policies have hard guarantees without the enforcement mechanism."),
        (["adversarial", "attack", "poisoning", "stealthy", "contested"], "adversarial autonomy and security-aware control", "M11 Adversarial Autonomy", "adversarial machine learning cyber physical systems stealthy actuation attack", "Do not infer attacker capability beyond the stated scenario."),
        (["dataset shift", "distribution shift", "ood", "unpredictable"], "distribution shift and operational uncertainty", "M05 Dataset Shift", "out of distribution detection robotics system level OOD runtime monitoring", "Do not treat OOD detection as a guarantee of safe recovery."),
        (["anomaly", "fault", "intrusion"], "fault/anomaly localization in autonomous systems", "M04 Anomaly and Fault Detection", "anomaly fault detection time series autonomous systems", "Do not assume anomaly detection identifies root cause by itself."),
        (["multi-agent", "fleet", "swarm", "cooperative", "distributed"], "distributed multi-agent coordination and validation", "M14 Multi-Agent and Cooperative Systems", "multi agent cooperative control fleet validation consensus communication constraints", "Do not reduce fleet assurance to single-agent safety."),
        (["privacy", "disclosure", "zero-knowledge", "zkp", "attestation"], "selective-disclosure assurance", "M03 Safety, Security, and Privacy", "privacy preserving verification attestation autonomous systems auditability", "Do not claim privacy mechanisms prove physical safety."),
        (["human", "driver", "operator", "governance", "accountability"], "human oversight, accountability, and governance", "M10 Policy and Governance", "human oversight accountability trustworthy AI autonomous systems governance", "Do not treat governance language as a technical control."),
        (["operational design domain", "odd", "level 3", "level 4", "level 5"], "operational design domain as the boundary of assurance", "M10 Policy and Governance", "operational design domain automated vehicles safety assurance", "Do not confuse higher autonomy level with higher assurance."),
        (["hybrid", "dynamics", "control", "feedback", "reachability"], "hybrid-system control and reachability assurance", "M06 Verification and Validation", "hybrid systems reachability neural network controller verification", "Do not assume reachability models capture perception uncertainty unless modeled."),
    ]
    concepts: list[str] = []
    modules: list[dict[str, str]] = []
    expansions: list[str] = []
    warnings: list[str] = []
    for triggers, concept, module, expansion, warning in rules:
        if any(t in q for t in triggers):
            concepts.append(concept)
            modules.append({"module": module, "reason": f"Triggered by: {', '.join(t for t in triggers if t in q)}"})
            expansions.append(expansion)
            warnings.append(warning)
    if not concepts:
        concepts = ["assurance case construction for learning-enabled cyber-physical systems"]
        modules = [{"module": "M01-02 Introduction to Assured Autonomy", "reason": "Generic assured-autonomy framing"}]
        expansions = ["learning enabled cyber physical systems assurance safety reliability"]
        warnings = ["Do not over-map a generic external anchor to a specific course module without evidence."]
    hidden_witnesses: list[str] = []
    behavioral_claims: list[str] = []
    proof_or_evidence_claims: list[str] = []
    escalation_points: list[str] = []
    physical_assumptions: list[str] = []
    assurance_gaps: list[str] = []
    trust_questions: list[str] = []
    verifier_questions: list[str] = []
    if any(t in q for t in ["zero-knowledge", "zkp", "proof", "witness", "attestation"]):
        hidden_witnesses.append("private witness or artifact that supports a public assurance claim")
        proof_or_evidence_claims.append("proof verifies an encoded statement, not the whole cyber-physical system")
        verifier_questions.append("What exact statement is checked, and what remains outside the proof?")
    if any(t in q for t in ["motion", "planning", "route", "trajectory", "vehicle", "robot", "autonomous"]):
        behavioral_claims.append("planner or controller output constrains physical behavior")
        escalation_points.append("cyber artifact becomes cyber-physical when it governs route, trajectory, actuation, or runtime control")
        physical_assumptions.append("sensing, dynamics, maps, timing, environment state, and actuator behavior match the model")
        assurance_gaps.append("runtime conformance, disturbance handling, perception validity, and simulator-to-real transfer remain unproven")
        trust_questions.append("What evidence from scenario X justifies limited trust on scenario Y?")
    summary = (
        "Heuristic lift: maps the external anchor to course-level assurance concepts "
        "using trigger phrases because the LLM lift produced no parseable visible JSON."
    )
    return _normalise_lift(
        {
            "literal_terms": literal_terms,
            "lifted_concepts": concepts,
            "hidden_witnesses": hidden_witnesses,
            "behavioral_claims": behavioral_claims,
            "proof_or_evidence_claims": proof_or_evidence_claims,
            "cyber_to_cps_escalation_points": escalation_points,
            "physical_world_assumptions": physical_assumptions,
            "assurance_gaps": assurance_gaps,
            "trust_transfer_questions": trust_questions,
            "verifier_questions": verifier_questions,
            "toy_model_interpretations": [],
            "possible_le_cps_instantiations": [],
            "threat_models": [],
            "module_hypotheses": modules,
            "retrieval_expansions": expansions,
            "what_not_to_overclaim": sorted(set(warnings)),
            "bridge_summary": summary,
        },
        raw,
        attempts=attempts,
    )


def seed_bridge_cards(bridge_dir: Path = DEFAULT_BRIDGE_DIR, *, overwrite: bool = False) -> list[Path]:
    """Create the first curated hidden bridge cards."""
    cards = [
        {
            "id": "zkp-agent-motion-planning",
            "title": "Designing Zero-Knowledge Proofs for Agent Motion Planning",
            "kind": "external_toy_problem",
            "source_url": "https://www.cee.org/sites/default/files/rsi/Papers/zhangceline_193939_5453198_Zhang_Celine_Sendova_Final.pdf",
            "source_context": "Regeneron STS finalist project; toy formal model for proof-bearing motion planning.",
            "bridge_concepts": [
                "proof-carrying autonomy",
                "selective-disclosure assurance",
                "privacy-preserving verification",
                "private witness public safety claim",
                "verifiable motion planning",
                "assurance under constrained disclosure",
            ],
            "abstract_assurance_principle": (
                "An autonomous agent can prove that a motion-planning solution satisfies "
                "a bounded property without revealing the plan, route, map, or witness."
            ),
            "toy_problem_role": (
                "A compact mathematical setting that isolates the trust structure of "
                "motion planning: private solution, public claim, verifier, bounded property."
            ),
            "modules": [
                "M03 Safety, Security, and Privacy",
                "M06 Verification and Validation",
                "M10 Policy and Governance",
                "M11 Adversarial Autonomy",
                "M12 Runtime Assurance",
                "M14 Multi-Agent and Cooperative Systems",
            ],
            "autonomy_instantiations": [
                "Autonomous vehicle proves a trajectory avoids forbidden regions without revealing the route.",
                "Fleet agent proves bounded reachability without revealing mission intent.",
                "Regulator or auditor verifies policy compliance without accessing proprietary planner internals.",
                "Runtime assurance monitor checks a proof artifact before accepting a planned trajectory.",
            ],
            "threat_models": [
                "Compromised or malicious planner proposes a physically unsafe but superficially plausible route.",
                "Untrusted fleet participant claims compliance while hiding its exact plan.",
                "Verifier needs assurance but full route disclosure would leak passenger, mission, or tactical information.",
            ],
            "retrieval_expansions": [
                "motion planning formal verification autonomous driving",
                "runtime assurance motion planner Simplex safe controller",
                "privacy autonomous systems route location confidentiality",
                "adversarial autonomy compromised planner cyber physical attack",
                "auditability proof artifact governance autonomous systems",
            ],
            "what_not_to_claim": [
                "Does not prove full autonomous-driving safety.",
                "Does not replace runtime monitoring or fallback control.",
                "Does not handle perception uncertainty or compromised inputs by itself.",
                "Does not prove the planner's intent or that the world model is accurate.",
            ],
        },
        {
            "id": "differential-privacy-assurance",
            "title": "Differential Privacy as Selective-Disclosure Assurance",
            "kind": "bridge_concept",
            "source_context": "Curated bridge from privacy theory to autonomous-system audit and telemetry.",
            "bridge_concepts": [
                "privacy-preserving assurance",
                "formal leakage bound",
                "aggregate telemetry without individual disclosure",
                "assurance under constrained disclosure",
                "auditability without raw-data exposure",
            ],
            "abstract_assurance_principle": (
                "A system can disclose useful aggregate evidence about behavior while "
                "formally limiting what is leaked about individuals, routes, events, or records."
            ),
            "modules": [
                "M03 Safety, Security, and Privacy",
                "M05 Dataset Shift",
                "M10 Policy and Governance",
                "M12 Runtime Assurance",
            ],
            "autonomy_instantiations": [
                "Fleet operator releases aggregate disengagement or near-miss statistics without exposing individual trips.",
                "Runtime monitor reports population-level safety telemetry while protecting passenger/location privacy.",
                "Auditor receives shift/anomaly summaries with bounded contribution from any one vehicle or person.",
            ],
            "retrieval_expansions": [
                "privacy autonomous systems data disclosure transparency",
                "governance auditability accountability AI systems",
                "dataset shift monitoring telemetry aggregate metrics",
                "runtime monitoring autonomous systems logs",
            ],
            "what_not_to_claim": [
                "Differential privacy does not prove route safety.",
                "Aggregate privacy does not solve individual runtime control.",
                "Privacy budgets and utility tradeoffs must be specified.",
            ],
        },
    ]
    out_paths: list[Path] = []
    bridge_dir.mkdir(parents=True, exist_ok=True)
    for card in cards:
        sub = bridge_dir / "_hidden_bridge_examples" / _slug(card["id"])
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / "card.json"
        if path.exists() and not overwrite:
            out_paths.append(path)
            continue
        path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        out_paths.append(path)
    return out_paths


def load_bridge_cards(bridge_dir: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if not bridge_dir.exists():
        return cards
    for path in sorted(bridge_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["_path"] = str(path)
        cards.append(data)
    return cards


def build_graph(cards: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, BridgeNode] = {}
    edges: list[BridgeEdge] = []

    def add_node(node_id: str, kind: str, label: str, description: str = "",
                 modules: list[str] | None = None, source: str = "curated") -> None:
        nodes.setdefault(
            node_id,
            BridgeNode(node_id, kind, label, description, modules or [], source),
        )

    for card in cards:
        card_id = str(card.get("id") or _slug(card.get("title", "card")))
        add_node(
            f"paper:{card_id}",
            str(card.get("kind", "bridge_card")),
            str(card.get("title", card_id)),
            str(card.get("abstract_assurance_principle", "")),
            list(card.get("modules", [])),
            str(card.get("source_context", "curated")),
        )
        for concept in card.get("bridge_concepts", []):
            cid = f"concept:{_slug(str(concept))}"
            add_node(cid, "bridge_concept", str(concept))
            edges.append(BridgeEdge(f"paper:{card_id}", "abstracts_to", cid))
        for module in card.get("modules", []):
            mid = f"module:{_slug(str(module).split()[0])}"
            add_node(mid, "module", str(module))
            edges.append(BridgeEdge(f"paper:{card_id}", "bridges_to", mid))
        for threat in card.get("threat_models", []):
            tid = f"threat:{_slug(str(threat))[:80]}"
            add_node(tid, "threat_model", str(threat))
            edges.append(BridgeEdge(f"paper:{card_id}", "motivated_by", tid))
        for item in card.get("autonomy_instantiations", []):
            iid = f"instantiation:{_slug(str(item))[:80]}"
            add_node(iid, "autonomy_instantiation", str(item))
            edges.append(BridgeEdge(f"paper:{card_id}", "instantiates_as", iid))
        for warning in card.get("what_not_to_claim", []):
            wid = f"limit:{_slug(str(warning))[:80]}"
            add_node(wid, "limit", str(warning))
            edges.append(BridgeEdge(f"paper:{card_id}", "limited_by", wid))

    return {
        "nodes": [asdict(n) for n in nodes.values()],
        "edges": [asdict(e) for e in edges],
        "cards": cards,
    }


def lift_query(query: str, *, provider: str, model: str | None, max_tokens: int) -> dict[str, Any]:
    llm = get_llm(provider, model=model)
    module_catalog = "\n".join(f"- {mid}: {name}" for mid, name in AA_MODULES.items())
    system = (
        "You are the conceptual preprocessor for an Assured Autonomy Bridge-Graph RAG. "
        "Your job is not to answer. Lift the user's literal query into abstraction-preserving "
        "bridges: toy problem role, hidden assurance principle, threat models, module links, "
        "retrieval expansions, and overclaim warnings. Explicitly separate these validity layers: "
        "cryptographic/formal validity, encoded-model validity, cyber-physical validity, runtime "
        "conformance, and trust transfer to new tasks. Return only a single compact JSON object. "
        "Use short strings. Do not use markdown. Use only the canonical course modules supplied "
        "by the user; do not invent module names."
    )
    user = (
        "User query:\n"
        f"{query}\n\n"
        "Canonical course modules. Use these exact IDs and names in module_hypotheses:\n"
        f"{module_catalog}\n\n"
        "Return exactly this JSON shape, filling every key:\n"
        "{\n"
        '  "literal_terms": ["..."],\n'
        '  "lifted_concepts": ["higher-order concepts, not synonyms"],\n'
        '  "hidden_witnesses": ["private object, state, plan, policy, data, or code artifact whose properties are asserted"],\n'
        '  "behavioral_claims": ["what the artifact claims or implies about agent/system behavior"],\n'
        '  "proof_or_evidence_claims": ["what is actually proved, tested, measured, or certified"],\n'
        '  "cyber_to_cps_escalation_points": ["where a cyber/software/formal artifact begins constraining physical behavior"],\n'
        '  "physical_world_assumptions": ["sensing, timing, dynamics, maps, environment, human, actuator, or deployment assumptions"],\n'
        '  "assurance_gaps": ["what remains unverified or untested"],\n'
        '  "trust_transfer_questions": ["what must hold for success on X to justify trust on Y"],\n'
        '  "verifier_questions": ["questions an auditor or assurance engineer should ask"],\n'
        '  "toy_model_interpretations": ["..."],\n'
        '  "possible_le_cps_instantiations": ["..."],\n'
        '  "threat_models": ["..."],\n'
        '  "module_hypotheses": [{"module": "M06: Verification and Validation", "reason": "..."}],\n'
        '  "retrieval_expansions": ["concrete search string"],\n'
        '  "what_not_to_overclaim": ["..."],\n'
        '  "bridge_summary": "one concise paragraph"\n'
        "}\n"
    )
    budgets = [max_tokens]
    for budget in [max(max_tokens * 2, 5000), max(max_tokens * 4, 9000)]:
        if budget not in budgets:
            budgets.append(budget)
    attempts: list[dict[str, Any]] = []
    last_raw = ""
    for budget in budgets:
        raw = llm.complete(system, user, temperature=0.1, max_tokens=budget)
        last_raw = raw
        data = _json_from_text(raw)
        attempts.append({
            "max_tokens": budget,
            "raw_chars": len(raw or ""),
            "parsed": bool(data),
            "raw_preview": (raw or "")[:400],
        })
        if data:
            return _normalise_lift(data, raw, attempts=attempts)
    return _heuristic_lift(query, raw=last_raw, attempts=attempts)


def score_cards(query: str, lift: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = " ".join([
        query,
        " ".join(map(str, lift.get("lifted_concepts", []))),
        " ".join(map(str, lift.get("hidden_witnesses", []))),
        " ".join(map(str, lift.get("behavioral_claims", []))),
        " ".join(map(str, lift.get("proof_or_evidence_claims", []))),
        " ".join(map(str, lift.get("cyber_to_cps_escalation_points", []))),
        " ".join(map(str, lift.get("physical_world_assumptions", []))),
        " ".join(map(str, lift.get("assurance_gaps", []))),
        " ".join(map(str, lift.get("trust_transfer_questions", []))),
        " ".join(map(str, lift.get("verifier_questions", []))),
        " ".join(map(str, lift.get("threat_models", []))),
        " ".join(map(str, lift.get("retrieval_expansions", []))),
    ]).lower()
    q_terms = set(re.findall(r"[a-z][a-z0-9-]{3,}", q))
    scored: list[dict[str, Any]] = []
    for card in cards:
        hay = json.dumps(card, ensure_ascii=False).lower()
        terms = set(re.findall(r"[a-z][a-z0-9-]{3,}", hay))
        overlap = sorted(q_terms & terms)
        concept_hits = [
            c for c in card.get("bridge_concepts", [])
            if str(c).lower() in q or any(tok in q_terms for tok in re.findall(r"[a-z][a-z0-9-]{3,}", str(c).lower()))
        ]
        score = len(overlap) + 3 * len(concept_hits)
        if score > 0:
            item = dict(card)
            item["_score"] = score
            item["_overlap"] = overlap[:30]
            item["_concept_hits"] = concept_hits
            scored.append(item)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def expanded_query(query: str, lift: dict[str, Any], matched_cards: list[dict[str, Any]]) -> str:
    parts = [query.strip(), "\n\nConceptual lift:"]
    for key in [
        "bridge_summary",
        "lifted_concepts",
        "hidden_witnesses",
        "behavioral_claims",
        "proof_or_evidence_claims",
        "cyber_to_cps_escalation_points",
        "physical_world_assumptions",
        "assurance_gaps",
        "trust_transfer_questions",
        "verifier_questions",
        "toy_model_interpretations",
        "possible_le_cps_instantiations",
        "threat_models",
        "retrieval_expansions",
        "what_not_to_overclaim",
    ]:
        val = lift.get(key)
        if val:
            parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")
    if matched_cards:
        parts.append("\nMatched hidden bridge cards:")
        for card in matched_cards[:5]:
            payload = {
                "title": card.get("title"),
                "bridge_concepts": card.get("bridge_concepts"),
                "abstract_assurance_principle": card.get("abstract_assurance_principle"),
                "modules": card.get("modules"),
                "retrieval_expansions": card.get("retrieval_expansions"),
                "what_not_to_claim": card.get("what_not_to_claim"),
            }
            parts.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(parts)


def synthesize_bridge_answer(query: str, lift: dict[str, Any], matched_cards: list[dict[str, Any]],
                             rec_md: str, *, provider: str, model: str | None,
                             max_tokens: int) -> str:
    llm = get_llm(provider, model=model)
    system = (
        "You answer as an Assured Autonomy Bridge-Graph RAG. Use the conceptual lift, "
        "hidden bridge cards, and retrieved course recommendations. Be explicit about "
        "higher-order bridges that simple keyword RAG would miss. Do not overclaim. "
        "Keep hidden bridge cards separate from course-corpus evidence."
    )
    user = (
        f"Original query:\n{query}\n\n"
        f"Conceptual lift:\n{json.dumps(lift, indent=2, ensure_ascii=False)}\n\n"
        f"Matched hidden bridge cards:\n{json.dumps(matched_cards[:5], indent=2, ensure_ascii=False)}\n\n"
        f"Course corpus recommendations:\n{rec_md}\n\n"
        "Answer requirements:\n"
        "- Start with the verdict.\n"
        "- Name the bridge concept(s), not only modules.\n"
        "- Explain what the hidden bridge cards added.\n"
        "- Identify course modules/readings that now become relevant.\n"
        "- Do not invent quotes, page claims, or paper-specific findings. If you cite a reading, "
        "use only the reading title/module and the retrieved recommendation rationale/passages above.\n"
        "- If a hidden bridge card supplies the idea rather than the course corpus, say so explicitly.\n"
        "- Include what not to claim.\n"
    )
    budgets = [max_tokens, max(max_tokens * 2, 4000), max(max_tokens * 3, 7000)]
    for budget in budgets:
        out = llm.complete(system, user, temperature=0.1, max_tokens=budget).strip()
        if out:
            return out
    return llm.complete(
        system,
        user + "\n\nReturn a compact answer in 8-12 bullets. Do not use a long preamble.",
        temperature=0.1,
        max_tokens=max(max_tokens * 2, 5000),
    ).strip()


def run_bridge_query(args: argparse.Namespace) -> int:
    bridge_dir = Path(args.bridge_dir)
    if args.init_seeds:
        seed_bridge_cards(bridge_dir, overwrite=args.overwrite_seeds)
    cards = load_bridge_cards(bridge_dir)
    graph = build_graph(cards)

    query = args.query or ""
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    if not query:
        out_dir = _unique_run_dir(Path(args.out_dir), "bridge_graph")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bridge_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote bridge graph: {out_dir / 'bridge_graph.json'}")
        print(f"Loaded {len(cards)} bridge cards.")
        return 0

    lift = lift_query(query, provider=args.provider, model=args.model, max_tokens=args.lift_max_tokens)
    matched = score_cards(query, lift, cards)
    eq = expanded_query(query, lift, matched[:args.max_bridge_cards])
    recs = recommend_corpus(
        eq,
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        top_papers=args.top_papers,
        passages_per_paper=args.passages_per_paper,
        candidate_chunks=args.candidate_chunks,
        include_optional=args.include_optional,
        reranker_model=None if args.no_reranker else args.reranker_model,
    )
    rec_md = render_recs(recs)
    answer = ""
    if not args.no_synthesis:
        answer = synthesize_bridge_answer(
            query,
            lift,
            matched[:args.max_bridge_cards],
            rec_md,
            provider=args.provider,
            model=args.model,
            max_tokens=args.synthesis_max_tokens,
        )

    out_dir = _unique_run_dir(Path(args.out_dir), "bridge_rag")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "provider": args.provider,
        "model": args.model,
        "lift": lift,
        "matched_bridge_cards": matched[:args.max_bridge_cards],
        "expanded_query": eq,
        "bridge_graph": graph,
        "corpus_recommendations": recs.to_dict(),
        "answer": answer,
    }
    (out_dir / "bridge_rag.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_bridge_markdown(payload, rec_md)
    (out_dir / "bridge_rag.md").write_text(md, encoding="utf-8")
    if answer:
        _safe_print(answer)
    else:
        _safe_print(md)
    _safe_print(f"\nSaved bridge RAG directory: {out_dir}")
    return 0


def render_bridge_markdown(payload: dict[str, Any], rec_md: str) -> str:
    lines = [
        "# Bridge-Graph RAG Run",
        "",
        "## Query",
        payload["query"],
        "",
        "## Conceptual Lift",
        "```json",
        json.dumps({k: v for k, v in payload["lift"].items() if k != "raw_model_output"}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Matched Hidden Bridge Cards",
    ]
    for card in payload["matched_bridge_cards"]:
        lines.append(f"- **{card.get('title')}** score={card.get('_score')}")
        lines.append(f"  - Concepts: {', '.join(map(str, card.get('bridge_concepts', [])))}")
        lines.append(f"  - Modules: {', '.join(map(str, card.get('modules', [])))}")
    lines.extend([
        "",
        "## Expanded Retrieval Query",
        "```text",
        payload["expanded_query"],
        "```",
        "",
        "## Answer",
        payload.get("answer") or "(synthesis disabled)",
        "",
        "## Course Recommendations",
        rec_md,
    ])
    return "\n".join(lines)


def _unique_run_dir(base: Path, prefix: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = base / f"{prefix}_{stamp}"
    n = 1
    while path.exists():
        path = base / f"{prefix}_{stamp}_{n}"
        n += 1
    return path


def _safe_print(text: str) -> None:
    import sys

    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(str(text).encode(enc, errors="replace").decode(enc, errors="replace"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--query-file")
    p.add_argument("--bridge-dir", default=str(DEFAULT_BRIDGE_DIR))
    p.add_argument("--init-seeds", action="store_true")
    p.add_argument("--overwrite-seeds", action="store_true")
    p.add_argument("--provider", default="mistral", choices=["mistral", "openai", "anthropic", "ollama", "stub", "peft-local"])
    p.add_argument("--model", default="mistral-large-latest")
    p.add_argument("--index-dir", default=".aate_index")
    p.add_argument("--papers-dir", default='papers')
    p.add_argument("--top-papers", type=int, default=14)
    p.add_argument("--passages-per-paper", type=int, default=2)
    p.add_argument("--candidate-chunks", type=int, default=120)
    p.add_argument("--max-bridge-cards", type=int, default=6)
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--lift-max-tokens", type=int, default=2200)
    p.add_argument("--synthesis-max-tokens", type=int, default=3500)
    p.add_argument("--no-synthesis", action="store_true")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    return run_bridge_query(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

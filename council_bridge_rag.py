"""Council Bridge RAG: multi-model bridge retrieval with scout/critic/synthesis.

This is the first incorporation of the multi-LLM RAG patterns we identified:
router-style conceptual lifting, mixture/council-style model diversity, and a
critic pass for overclaim control. It deliberately reuses Bridge-Graph RAG's
retrieval machinery so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .bridge_graph_rag import (
    DEFAULT_BRIDGE_DIR,
    build_graph,
    expanded_query,
    lift_query,
    load_bridge_cards,
    render_bridge_markdown,
    score_cards,
    seed_bridge_cards,
)
from .corpus_recommender import recommend_corpus, render_markdown as render_recs
from .llm import get_llm
from .concept_graph import ConceptGraph, query_graph, render_query_markdown


DEFAULT_OUT_DIR = Path("council_bridge_runs")


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def _unique_run_dir(base: Path, prefix: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return base / f"{prefix}_{stamp}"


def _model_label(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _complete_nonempty(llm: Any, system: str, user: str, *, max_tokens: int,
                       temperature: float = 0.1) -> str:
    """Call an LLM and retry empty visible outputs with larger budgets.

    Some reasoning models can consume the requested completion budget before
    emitting visible text. Empty output is not useful in a saved RAG run, so we
    retry with larger budgets and finally ask for a compact answer.
    """
    budgets = [max_tokens, max(max_tokens * 2, 4000), max(max_tokens * 3, 7000)]
    for budget in budgets:
        out = llm.complete(system, user, temperature=temperature, max_tokens=budget).strip()
        if out:
            return out
    compact_user = (
        user
        + "\n\nThe previous attempts returned no visible text. Return a compact final answer "
        "in 8-12 bullets. Do not use long preambles."
    )
    return llm.complete(
        system,
        compact_user,
        temperature=temperature,
        max_tokens=max(max_tokens * 2, 5000),
    ).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()))


def _short_recommendation_summary(recs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in getattr(recs, "recommendations", []):
        rows.append({
            "rank": rec.rank,
            "filename": rec.filename,
            "module_id": rec.module_id,
            "module_name": rec.module_name,
            "required": rec.required,
            "relevance_score": rec.relevance_score,
            "diversity_adjusted_score": rec.diversity_adjusted_score,
            "rationale": rec.rationale,
            "passage_count": len(rec.passages),
        })
    return rows


def _evaluate_evidence_gate(query: str, lift: dict[str, Any], recs: Any) -> dict[str, Any]:
    """Deterministic CRAG-style evidence sufficiency check."""
    recommendations = getattr(recs, "recommendations", [])
    routed_modules = getattr(recs, "routed_modules", [])
    rec_modules = {rec.module_id for rec in recommendations}
    routed_top = {m["module_id"] for m in routed_modules[:6]}
    raw_top_scores = [float(getattr(rec, "relevance_score", 0.0) or 0.0) for rec in recommendations[:5]]
    score_outliers = [s for s in raw_top_scores if s > 5.0]
    top_scores = [min(s, 1.0) for s in raw_top_scores]
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    required_count = sum(1 for rec in recommendations if getattr(rec, "required", False))
    passage_count = sum(len(getattr(rec, "passages", [])) for rec in recommendations)

    concept_terms = _tokens(" ".join(map(str, lift.get("lifted_concepts", []))))
    for key in [
        "hidden_witnesses",
        "behavioral_claims",
        "proof_or_evidence_claims",
        "cyber_to_cps_escalation_points",
        "physical_world_assumptions",
        "assurance_gaps",
        "trust_transfer_questions",
        "verifier_questions",
    ]:
        concept_terms |= _tokens(" ".join(map(str, lift.get(key, []))))
    concept_terms |= _tokens(" ".join(map(str, lift.get("threat_models", []))))
    concept_terms |= _tokens(" ".join(map(str, lift.get("possible_le_cps_instantiations", []))))
    rec_text = " ".join(
        [
            getattr(rec, "filename", "")
            + " "
            + getattr(rec, "module_name", "")
            + " "
            + getattr(rec, "rationale", "")
            + " "
            + " ".join(getattr(p, "text", "") for p in getattr(rec, "passages", []))
            for rec in recommendations
        ]
    )
    rec_terms = _tokens(rec_text)
    covered_terms = sorted(term for term in concept_terms if term in rec_terms)
    coverage_ratio = len(covered_terms) / max(len(concept_terms), 1)

    warnings: list[str] = []
    if len(recommendations) < 4:
        warnings.append("fewer than four recommended papers")
    if len(rec_modules) < 3:
        warnings.append("low module diversity")
    if routed_top and len(routed_top & rec_modules) < min(3, len(routed_top)):
        warnings.append("retrieved modules cover too few routed modules")
    if avg_top_score < 0.35:
        warnings.append("low average top relevance score")
    if score_outliers:
        warnings.append("retrieval score-scale outlier; inspect top-ranked evidence")
    if required_count < max(2, len(recommendations) // 3):
        warnings.append("few required readings in evidence set")
    if passage_count < len(recommendations):
        warnings.append("some recommendations have no passages")
    if coverage_ratio < 0.12:
        warnings.append("low overlap between lifted concepts and retrieved evidence text")

    status = "strong"
    if warnings:
        status = "usable"
    if len(warnings) >= 3 or not recommendations:
        status = "weak"

    missing_modules = [
        {"module_id": m["module_id"], "module_name": m["module_name"], "score": m["score"]}
        for m in routed_modules[:6]
        if m["module_id"] not in rec_modules
    ]
    uncovered_concepts = [
        c for c in lift.get("lifted_concepts", [])
        if not (_tokens(str(c)) & rec_terms)
    ][:10]
    expansion_terms = [
        str(c) for c in (
            uncovered_concepts[:6]
            + [m["module_name"] for m in missing_modules[:4]]
            + lift.get("assurance_gaps", [])[:4]
            + lift.get("cyber_to_cps_escalation_points", [])[:4]
            + lift.get("retrieval_expansions", [])[:4]
        )
        if str(c).strip()
    ]
    return {
        "status": status,
        "warnings": warnings,
        "avg_top_relevance": round(avg_top_score, 4),
        "raw_top_relevance_scores": [round(s, 4) for s in raw_top_scores],
        "score_outliers": [round(s, 4) for s in score_outliers],
        "recommended_papers": len(recommendations),
        "recommended_modules": sorted(rec_modules),
        "routed_top_modules": sorted(routed_top),
        "required_count": required_count,
        "passage_count": passage_count,
        "lifted_concept_coverage_ratio": round(coverage_ratio, 4),
        "covered_lifted_terms_sample": covered_terms[:30],
        "missing_routed_modules": missing_modules,
        "uncovered_lifted_concepts": uncovered_concepts,
        "suggested_expansion_terms": expansion_terms[:12],
        "recommendation_summary": _short_recommendation_summary(recs),
    }


def _expand_for_evidence_gap(query: str, lift: dict[str, Any], gate: dict[str, Any]) -> str:
    parts = [
        query,
        "\n\nCRAG evidence-gap expansion:",
        "Prioritize retrieved passages that directly address these missing or weak concepts:",
    ]
    for term in gate.get("suggested_expansion_terms", [])[:12]:
        parts.append(f"- {term}")
    parts.append("\nUse these lifted bridge concepts as retrieval anchors:")
    for concept in lift.get("lifted_concepts", [])[:12]:
        parts.append(f"- {concept}")
    return "\n".join(parts)


def _scout_answer(
    *,
    query: str,
    lift: dict[str, Any],
    matched_cards: list[dict[str, Any]],
    rec_md: str,
    concept_graph_md: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> str:
    llm = get_llm(provider, model=model)
    system = (
        "You are the scout model in a multi-model Assured Autonomy RAG council. "
        "Your job is broad discovery: identify promising bridges, risks, and "
        "course-corpus hooks. Be concrete and cite only retrieved course readings "
        "by title/module/path from the provided recommendations. Mark speculative "
        "ideas as speculative. Use the abstraction lift as an engineering ledger: "
        "hidden witness, behavioral claim, proof/evidence claim, cyber-to-CPS "
        "escalation point, physical assumptions, assurance gaps, and trust-transfer "
        "questions."
    )
    user = (
        f"Original query:\n{query}\n\n"
        f"Conceptual lift:\n{json.dumps(lift, indent=2, ensure_ascii=False)}\n\n"
        f"Matched hidden bridge cards:\n{json.dumps(matched_cards[:6], indent=2, ensure_ascii=False)}\n\n"
        f"Persistent concept graph context:\n{concept_graph_md or '(none)'}\n\n"
        f"Retrieved course recommendations:\n{rec_md}\n\n"
        "Produce a scout memo with:\n"
        "1. Candidate higher-order bridge concepts.\n"
        "2. The validity chain: proof/evidence -> model claim -> cyber-physical behavior -> runtime conformance.\n"
        "3. Strongest course-corpus connections.\n"
        "4. Implementation or research risks.\n"
        "5. What not to overclaim.\n"
        "6. Best next experiments or checks.\n"
    )
    return _complete_nonempty(
        llm, system, user, temperature=0.2, max_tokens=max_tokens,
    )


def _critic_answer(
    *,
    query: str,
    lift: dict[str, Any],
    matched_cards: list[dict[str, Any]],
    rec_md: str,
    concept_graph_md: str,
    scout: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> str:
    llm = get_llm(provider, model=model)
    system = (
        "You are the critic/judge in a multi-model Assured Autonomy RAG council. "
        "Your job is not to be imaginative. Your job is to find unsupported claims, "
        "module-mapping mistakes, weak analogies, missing tests, and places where "
        "hidden bridge cards were overused. Preserve good insights, but tighten them. "
        "Audit whether the answer wrongly collapses cryptographic/formal validity, "
        "encoded-model validity, cyber-physical validity, runtime conformance, and "
        "trust transfer."
    )
    user = (
        f"Original query:\n{query}\n\n"
        f"Conceptual lift:\n{json.dumps(lift, indent=2, ensure_ascii=False)}\n\n"
        f"Matched hidden bridge cards:\n{json.dumps(matched_cards[:6], indent=2, ensure_ascii=False)}\n\n"
        f"Persistent concept graph context:\n{concept_graph_md or '(none)'}\n\n"
        f"Retrieved course recommendations:\n{rec_md}\n\n"
        f"Scout answer to audit:\n{scout}\n\n"
        "Return a critic memo with:\n"
        "1. Supported insights to keep.\n"
        "2. Validity-layer collapses or unsupported leaps to remove.\n"
        "3. Missing Assured Autonomy concepts/readings/tests.\n"
        "4. Revised framing in one paragraph.\n"
        "5. Confidence notes.\n"
    )
    return _complete_nonempty(
        llm, system, user, temperature=0.1, max_tokens=max_tokens,
    )


def _final_synthesis(
    *,
    query: str,
    lift: dict[str, Any],
    matched_cards: list[dict[str, Any]],
    rec_md: str,
    concept_graph_md: str,
    scout: str,
    critic: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> str:
    llm = get_llm(provider, model=model)
    system = (
        "You are the final synthesizer for a multi-model Assured Autonomy RAG council. "
        "Use the scout for breadth, the critic for discipline, the conceptual lift for "
        "hidden bridges, and the retrieved course recommendations for evidence. Do not "
        "invent paper details. Separate course-corpus evidence from speculative bridge-card "
        "extensions. The final answer must preserve the validity chain instead of treating "
        "a proof, benchmark, or code artifact as a complete cyber-physical safety claim."
    )
    user = (
        f"Original query:\n{query}\n\n"
        f"Conceptual lift:\n{json.dumps(lift, indent=2, ensure_ascii=False)}\n\n"
        f"Matched hidden bridge cards:\n{json.dumps(matched_cards[:6], indent=2, ensure_ascii=False)}\n\n"
        f"Persistent concept graph context:\n{concept_graph_md or '(none)'}\n\n"
        f"Retrieved course recommendations:\n{rec_md}\n\n"
        f"Scout memo:\n{scout}\n\n"
        f"Critic memo:\n{critic}\n\n"
        "Write the final answer with:\n"
        "- Verdict.\n"
        "- Abstraction lift: hidden witness, behavioral claim, proof/evidence claim, physical assumptions.\n"
        "- Validity chain: what is proved/tested, what is modeled, what is physical, what must be monitored at runtime.\n"
        "- What the RAG learned that keyword search would miss.\n"
        "- Course-corpus grounding with titles/modules/local paths when available.\n"
        "- What remains speculative or unproven.\n"
        "- Trust-transfer questions: what would justify moving from X to Y.\n"
        "- Concrete next experiments/checks.\n"
        "- A short 'do not claim' section.\n"
    )
    return _complete_nonempty(
        llm, system, user, temperature=0.1, max_tokens=max_tokens,
    )


def render_council_markdown(payload: dict[str, Any], rec_md: str) -> str:
    bridge_payload = {
        "query": payload["query"],
        "lift": payload["lift"],
        "matched_bridge_cards": payload["matched_bridge_cards"],
        "expanded_query": payload["expanded_query"],
        "answer": payload.get("final_answer", ""),
    }
    base = render_bridge_markdown(bridge_payload, rec_md)
    return "\n".join([
        "# Council Bridge RAG Run",
        "",
        "## Model Roles",
        f"- Lift: `{payload['models']['lift']}`",
        f"- Scout: `{payload['models']['scout']}`",
        f"- Critic: `{payload['models']['critic']}`",
        f"- Synthesis: `{payload['models']['synthesis']}`",
        "",
        "## Evidence Gate",
        f"- Status: `{payload.get('evidence_gate', {}).get('status', 'unknown')}`",
        f"- Warnings: {', '.join(payload.get('evidence_gate', {}).get('warnings', [])) or 'none'}",
        f"- Corrective retrieval attempts: {len(payload.get('retrieval_attempts', []))}",
        f"- Concept graph context: {'yes' if payload.get('concept_graph_query') else 'no'}",
        "",
        base.replace("# Bridge-Graph RAG Run\n\n", ""),
        "",
        "## Scout Memo",
        payload.get("scout_answer") or "(not run)",
        "",
        "## Critic Memo",
        payload.get("critic_answer") or "(not run)",
    ])


def run_council_bridge_query(args: argparse.Namespace) -> int:
    bridge_dir = Path(args.bridge_dir)
    if args.init_seeds:
        seed_bridge_cards(bridge_dir, overwrite=args.overwrite_seeds)
    cards = load_bridge_cards(bridge_dir)
    bridge_graph = build_graph(cards)

    query = args.query or ""
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    if not query:
        print("council-bridge-rag requires a query or --query-file", flush=True)
        return 2

    lift = lift_query(
        query,
        provider=args.lift_provider,
        model=args.lift_model,
        max_tokens=args.lift_max_tokens,
    )
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
    evidence_gate = _evaluate_evidence_gate(query, lift, recs)
    retrieval_attempts = [
        {
            "attempt": 1,
            "expanded_query": eq,
            "evidence_gate": evidence_gate,
        }
    ]
    if args.corrective_retrieval and evidence_gate["status"] == "weak":
        corrective_query = _expand_for_evidence_gap(eq, lift, evidence_gate)
        recs = recommend_corpus(
            corrective_query,
            index_dir=Path(args.index_dir),
            papers_dir=Path(args.papers_dir),
            top_papers=args.top_papers,
            passages_per_paper=args.passages_per_paper,
            candidate_chunks=max(args.candidate_chunks, 100),
            include_optional=args.include_optional,
            reranker_model=None if args.no_reranker else args.reranker_model,
        )
        evidence_gate = _evaluate_evidence_gate(query, lift, recs)
        retrieval_attempts.append(
            {
                "attempt": 2,
                "expanded_query": corrective_query,
                "evidence_gate": evidence_gate,
            }
        )
    rec_md = render_recs(recs)
    concept_graph_md = ""
    concept_graph_query_result: dict[str, Any] | None = None
    if getattr(args, "concept_graph", None):
        cg_path = Path(args.concept_graph)
        if cg_path.exists():
            cg = ConceptGraph.from_dict(json.loads(cg_path.read_text(encoding="utf-8")))
            concept_graph_query_result = query_graph(cg, eq, top_k=args.concept_graph_top_k)
            concept_graph_md = render_query_markdown(concept_graph_query_result)

    if args.gate_only:
        scout = ""
        critic = ""
        final = ""
    else:
        scout = _scout_answer(
            query=query,
            lift=lift,
            matched_cards=matched[:args.max_bridge_cards],
            rec_md=rec_md,
            concept_graph_md=concept_graph_md,
            provider=args.scout_provider,
            model=args.scout_model,
            max_tokens=args.scout_max_tokens,
        )
        critic = ""
        if not args.no_critic:
            critic = _critic_answer(
                query=query,
                lift=lift,
                matched_cards=matched[:args.max_bridge_cards],
                rec_md=rec_md,
                concept_graph_md=concept_graph_md,
                scout=scout,
                provider=args.critic_provider,
                model=args.critic_model,
                max_tokens=args.critic_max_tokens,
            )
        final = ""
        if not args.no_synthesis:
            final = _final_synthesis(
                query=query,
                lift=lift,
                matched_cards=matched[:args.max_bridge_cards],
                rec_md=rec_md,
                concept_graph_md=concept_graph_md,
                scout=scout,
                critic=critic,
                provider=args.synthesis_provider,
                model=args.synthesis_model,
                max_tokens=args.synthesis_max_tokens,
            )

    out_dir = _unique_run_dir(Path(args.out_dir), "council_bridge_rag")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "models": {
            "lift": _model_label(args.lift_provider, args.lift_model),
            "scout": _model_label(args.scout_provider, args.scout_model),
            "critic": _model_label(args.critic_provider, args.critic_model),
            "synthesis": _model_label(args.synthesis_provider, args.synthesis_model),
        },
        "lift": lift,
        "matched_bridge_cards": matched[:args.max_bridge_cards],
        "expanded_query": eq,
        "bridge_graph": bridge_graph,
        "concept_graph_query": concept_graph_query_result,
        "evidence_gate": evidence_gate,
        "retrieval_attempts": retrieval_attempts,
        "corpus_recommendations": recs.to_dict(),
        "scout_answer": scout,
        "critic_answer": critic,
        "final_answer": final,
    }
    (out_dir / "council_bridge_rag.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = render_council_markdown(payload, rec_md)
    (out_dir / "council_bridge_rag.md").write_text(md, encoding="utf-8")

    _safe_print(final or critic or scout)
    _safe_print(f"\nSaved council bridge RAG directory: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--query-file")
    p.add_argument("--bridge-dir", default=str(DEFAULT_BRIDGE_DIR))
    p.add_argument("--init-seeds", action="store_true")
    p.add_argument("--overwrite-seeds", action="store_true")
    p.add_argument("--index-dir", default=".aate_index")
    p.add_argument("--papers-dir", default='papers')
    p.add_argument("--top-papers", type=int, default=10)
    p.add_argument("--passages-per-paper", type=int, default=1)
    p.add_argument("--candidate-chunks", type=int, default=90)
    p.add_argument("--max-bridge-cards", type=int, default=6)
    p.add_argument("--concept-graph",
                   help="Optional concept_graph.json built with `aate concept-graph build`.")
    p.add_argument("--concept-graph-top-k", type=int, default=18)
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--lift-provider", default="openai")
    p.add_argument("--lift-model", default="gpt-5.5")
    p.add_argument("--scout-provider", default="xai")
    p.add_argument("--scout-model", default="grok-4.20-reasoning")
    p.add_argument("--critic-provider", default="openai")
    p.add_argument("--critic-model", default="gpt-4.1")
    p.add_argument("--synthesis-provider", default="openai")
    p.add_argument("--synthesis-model", default="gpt-5.5")
    p.add_argument("--lift-max-tokens", type=int, default=2600)
    p.add_argument("--scout-max-tokens", type=int, default=2600)
    p.add_argument("--critic-max-tokens", type=int, default=2200)
    p.add_argument("--synthesis-max-tokens", type=int, default=3200)
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--no-synthesis", action="store_true")
    p.add_argument("--no-corrective-retrieval", dest="corrective_retrieval",
                   action="store_false", default=True)
    p.add_argument("--gate-only", action="store_true",
                   help="Run lift, retrieval, evidence gate, and corrective retrieval only.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    return run_council_bridge_query(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

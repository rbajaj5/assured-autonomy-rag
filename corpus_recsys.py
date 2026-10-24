"""Recommendation-system layer over the Assured Autonomy corpus.

This command is deliberately different from the normal RAG answer. It treats
the corpus as a reading recommender: given a user question, answer briefly, then
recommend a small set of papers and offer a paper-centered discussion path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus_recommender import CorpusRecommendationReport, recommend_corpus
from .llm import get_llm


DEFAULT_OUT_DIR = Path("recsys_runs")


@dataclass
class RecSysConfig:
    index_dir: Path = Path(".aate_index")
    papers_dir: Path = Path('papers')
    provider: str = "mistral"
    model: str = "mistral-large-latest"
    top_papers: int = 6
    passages_per_paper: int = 2
    candidate_chunks: int = 90
    route_top_k: int = 6
    include_optional: bool = False
    max_tokens: int = 2200
    out_dir: Path = DEFAULT_OUT_DIR
    no_llm: bool = False


def load_key_file(path: str | Path | None, *, env_name: str) -> None:
    """Load an API key into the environment without printing it.

    This is the safe alternative to embedding an API key in a bundled EXE.
    """
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    key = p.read_text(encoding="utf-8").strip().strip('"').strip("'").strip()
    if key:
        os.environ[env_name] = key


def run_recsys(query: str, cfg: RecSysConfig) -> dict[str, Any]:
    recs = recommend_corpus(
        query,
        index_dir=cfg.index_dir,
        papers_dir=cfg.papers_dir,
        top_papers=cfg.top_papers,
        passages_per_paper=cfg.passages_per_paper,
        candidate_chunks=cfg.candidate_chunks,
        include_optional=cfg.include_optional,
        route_top_k=cfg.route_top_k,
    )
    synthesis = heuristic_synthesis(query, recs)
    if not cfg.no_llm:
        try:
            llm = get_llm(cfg.provider, model=cfg.model)
            synthesis = llm_synthesis(query, recs, llm, max_tokens=cfg.max_tokens)
        except Exception as exc:
            synthesis["llm_error"] = f"{type(exc).__name__}: {exc}"
            synthesis["llm_fallback_used"] = True
    result = {
        "query": query,
        "config": {
            **asdict(cfg),
            "index_dir": str(cfg.index_dir),
            "papers_dir": str(cfg.papers_dir),
            "out_dir": str(cfg.out_dir),
        },
        "recommendation_report": recs.to_dict(),
        "synthesis": synthesis,
    }
    return result


def llm_synthesis(query: str, recs: CorpusRecommendationReport, llm: Any, *, max_tokens: int) -> dict[str, Any]:
    source_block = "\n\n".join(
        [
            f"[R{rec.rank}] {rec.filename}\n"
            f"Module: {rec.module_id} {rec.module_name}; Required: {rec.required}; Local path: {rec.relpath}\n"
            f"Rationale: {rec.rationale}\n"
            + "\n".join(
                f"- {p.citation}: {p.text[:700]}"
                for p in rec.passages
            )
            for rec in recs.recommendations
        ]
    )
    system = (
        "You are a corpus-aware recommendation system for a JHU Assured Autonomy student. "
        "Answer the user's question briefly, then recommend specific course papers to read. "
        "Your job is not just to answer; it is to decide whether a paper-centered discussion "
        "would help the user learn or resolve the question."
    )
    user = f"""
User question:
{query}

Retrieved reading candidates:
{source_block}

Return strict JSON with these keys:
{{
  "short_answer": "2-5 sentence answer grounded in the retrieved readings",
  "recommended_discussion": {{
    "should_offer": true,
    "paper_rank": 1,
    "paper": "filename.pdf",
    "module": "Module name",
    "why_this_paper": "why this is the best paper to discuss first",
    "offer_text": "A natural one-sentence offer to the user asking if they are open to discussing this paper"
  }},
  "reading_path": [
    {{"rank": 1, "paper": "filename.pdf", "role": "why read it", "discussion_question": "one useful question"}}
  ],
  "assurance_angles": ["module-grounded angle 1", "module-grounded angle 2"],
  "overclaim_warning": "what not to infer from these readings"
}}
"""
    raw = llm.complete(system, user, temperature=0.1, max_tokens=max_tokens)
    data = _json_from_text(raw)
    if not data:
        data = heuristic_synthesis(query, recs)
        data["raw_model_output"] = raw
        data["parse_fallback_used"] = True
        return data
    data["raw_model_output"] = raw
    return data


def heuristic_synthesis(query: str, recs: CorpusRecommendationReport) -> dict[str, Any]:
    first = recs.recommendations[0] if recs.recommendations else None
    routed = ", ".join(
        f"{m['module_id']} {m['module_name']}" for m in recs.routed_modules[:3]
    ) or "the retrieved course modules"
    if not first:
        return {
            "short_answer": f"I could not retrieve a confident paper shortlist for: {query}",
            "recommended_discussion": {"should_offer": False},
            "reading_path": [],
            "assurance_angles": [],
            "overclaim_warning": "No corpus grounding was found.",
        }
    return {
        "short_answer": (
            f"This question routes mainly to {routed}. The corpus recommender found "
            f"{first.filename} as the strongest starting point, with supporting readings "
            "for adjacent assurance concepts."
        ),
        "recommended_discussion": {
            "should_offer": True,
            "paper_rank": first.rank,
            "paper": first.filename,
            "module": first.module_name,
            "why_this_paper": first.rationale,
            "offer_text": (
                f"Would you be open to discussing {first.filename} first? "
                f"It looks like the best entry point for this question in {first.module_name}."
            ),
        },
        "reading_path": [
            {
                "rank": rec.rank,
                "paper": rec.filename,
                "role": rec.rationale,
                "discussion_question": _discussion_question(rec.module_name, rec.filename),
            }
            for rec in recs.recommendations[:5]
        ],
        "assurance_angles": [m["module_name"] for m in recs.routed_modules[:5]],
        "overclaim_warning": (
            "A reading recommendation is not itself an assurance claim; use the paper to "
            "frame what evidence or test would be needed."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    syn = result["synthesis"]
    recs = result["recommendation_report"]["recommendations"]
    lines = [
        "# Corpus RecSys Answer",
        "",
        f"**Question:** {result['query']}",
        "",
        "## Short Answer",
        "",
        str(syn.get("short_answer", "")).strip(),
        "",
        "## Recommended Discussion",
        "",
    ]
    offer = syn.get("recommended_discussion", {}) or {}
    if offer.get("should_offer"):
        lines.extend([
            f"**Paper:** {offer.get('paper', '')}",
            "",
            f"**Module:** {offer.get('module', '')}",
            "",
            f"**Why:** {offer.get('why_this_paper', '')}",
            "",
            f"> {offer.get('offer_text', '')}",
            "",
        ])
    else:
        lines.append("No paper-centered discussion offer was generated.")
        lines.append("")

    lines.extend(["## Reading Path", ""])
    for item in syn.get("reading_path", []):
        lines.extend([
            f"### {item.get('rank')}. {item.get('paper')}",
            "",
            f"- Role: {item.get('role')}",
            f"- Discussion question: {item.get('discussion_question')}",
            "",
        ])

    lines.extend(["## Retrieved Papers", ""])
    for rec in recs:
        req = "Required" if rec.get("required") else "Optional/Supplementary"
        lines.extend([
            f"### {rec.get('rank')}. {rec.get('filename')}",
            "",
            f"- Module: {rec.get('module_id')} {rec.get('module_name')}",
            f"- Reading status: {req}",
            f"- Local path: {rec.get('relpath')}",
            f"- Why: {rec.get('rationale')}",
        ])
        for passage in rec.get("passages", []):
            lines.append(f"- Passage: {passage.get('citation')} score={passage.get('score')}")
        lines.append("")

    lines.extend([
        "## Assurance Angles",
        "",
        *[f"- {angle}" for angle in syn.get("assurance_angles", [])],
        "",
        "## Overclaim Warning",
        "",
        str(syn.get("overclaim_warning", "")).strip(),
        "",
    ])
    if syn.get("llm_error"):
        lines.extend(["## LLM Warning", "", str(syn["llm_error"]), ""])
    return "\n".join(lines)


def save_result(result: dict[str, Any], base_dir: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = base_dir / f"corpus_recsys_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_recsys.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "corpus_recsys.md").write_text(render_markdown(result), encoding="utf-8")
    return out


def _discussion_question(module_name: str, filename: str) -> str:
    return f"How would {filename} change the evidence we need for a {module_name} assurance claim?"


def _json_from_text(raw: str) -> dict[str, Any]:
    raw = raw.strip()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--query-file")
    parser.add_argument("--index-dir", default=".aate_index")
    parser.add_argument("--papers-dir", default='papers')
    parser.add_argument("--provider", default="mistral",
                        choices=["mistral", "openai", "xai", "grok", "anthropic", "ollama", "stub", "peft-local"])
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--mistral-key-file",
                        help="Read Mistral API key from a local file instead of embedding it in the executable")
    parser.add_argument("--top-papers", type=int, default=6)
    parser.add_argument("--passages-per-paper", type=int, default=2)
    parser.add_argument("--candidate-chunks", type=int, default=90)
    parser.add_argument("--route-top-k", type=int, default=6)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query = args.query
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    if not query:
        print("corpus-recsys requires a query or --query-file")
        return 2
    load_key_file(args.mistral_key_file, env_name="MISTRAL_API_KEY")
    cfg = RecSysConfig(
        index_dir=Path(args.index_dir),
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        top_papers=args.top_papers,
        passages_per_paper=args.passages_per_paper,
        candidate_chunks=args.candidate_chunks,
        route_top_k=args.route_top_k,
        include_optional=args.include_optional,
        max_tokens=args.max_tokens,
        out_dir=Path(args.out_dir),
        no_llm=args.no_llm,
    )
    result = run_recsys(query, cfg)
    out = save_result(result, cfg.out_dir)
    print(render_markdown(result))
    print(f"Saved Corpus RecSys run directory: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

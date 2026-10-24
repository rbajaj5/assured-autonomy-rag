"""Two-model debate over policy/source materials."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from .hybrid_retriever import HybridPaperRetriever, HybridRetrievalConfig
from .indexing import Hit, RagIndex
from .llm import get_llm
from .paths import DEFAULT_INDEX_DIR, PAPERS_DIR
from .policy_rag import _expand_policy_source_refs, _safe_name, build_multi_policy_prompt, load_policy_sources


@dataclass
class DebaterConfig:
    provider: str
    model: str
    role: str


@dataclass
class PolicyDebateConfig:
    debater_a: DebaterConfig
    debater_b: DebaterConfig
    judge_provider: str
    judge_model: str
    index_dir: Path = DEFAULT_INDEX_DIR
    papers_dir: Path = PAPERS_DIR
    aa_top_k_per_concept: int = 3
    max_policy_chars: int = 60000
    max_tokens: int = 5000
    stage_retries: int = 2
    out_dir: Path = Path("policy_runs")


DEFAULT_DEBATE_QUESTION = (
    "Can a JHU student realistically obtain JuReko-related data for academic research? "
    "Debate whether the best path is raw corpus access, a narrow subset, derived outputs, "
    "or CAL2Lab interface access. Focus on Germany and treat the British pilot as peripheral."
)


def run_policy_debate(*, cfg: PolicyDebateConfig,
                      policy_urls: list[str] | None = None,
                      policy_file: str | None = None,
                      policy_text: str | None = None,
                      question: str = DEFAULT_DEBATE_QUESTION) -> dict:
    run_dir, run_name = _new_run_dir(cfg)
    items = load_policy_sources(
        policy_urls=policy_urls,
        policy_file=policy_file,
        policy_text=policy_text,
    )
    source_prompt = build_multi_policy_prompt(
        items,
        question,
        max_policy_chars=cfg.max_policy_chars,
    )
    aa_evidence = _retrieve_aa_concepts(cfg, question)
    debate_prompt = (
        f"{source_prompt}\n\n"
        f"Assured Autonomy concept evidence:\n{aa_evidence['prompt_block']}\n\n"
        "Debate rule for this run:\n"
        "- Treat each Assured Autonomy concept as a reusable reasoning vector: "
        "name it, connect it to the policy facts, and explain how it changes "
        "the access strategy.\n"
        "- Do not merely cite the course readings. Argue from their concepts.\n"
        "- Use source-page citations as [S1], [S2], etc. Use course-concept "
        "citations as [AA1], [AA2], etc.\n"
    )
    a_client = get_llm(cfg.debater_a.provider, model=cfg.debater_a.model)
    b_client = get_llm(cfg.debater_b.provider, model=cfg.debater_b.model)
    judge_client = get_llm(cfg.judge_provider, model=cfg.judge_model)

    result = {
        "question": question,
        "policy_source": [
            {
                "source_kind": item.source_kind,
                "source": item.source,
                "title": item.title,
                "text_len": len(item.text),
            }
            for item in items
        ],
        "assured_autonomy_evidence": aa_evidence["items"],
        "debater_a": cfg.debater_a.__dict__,
        "debater_b": cfg.debater_b.__dict__,
        "judge": {"provider": cfg.judge_provider, "model": cfg.judge_model},
        "a_open": "",
        "b_open": "",
        "a_rebuttal": "",
        "b_rebuttal": "",
        "judgment": "",
        "status": "started",
    }
    _save_progress(result, run_dir, run_name)

    a_open = _call_stage(
        lambda budget: _ask_debater(a_client, cfg.debater_a, debate_prompt, question, max_tokens=budget),
        stage="a_open",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["a_open"] = _expand_policy_source_refs(a_open, items)
    result["status"] = "a_open_complete"
    _save_progress(result, run_dir, run_name)

    b_open = _call_stage(
        lambda budget: _ask_debater(b_client, cfg.debater_b, debate_prompt, question, max_tokens=budget),
        stage="b_open",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["b_open"] = _expand_policy_source_refs(b_open, items)
    result["status"] = "b_open_complete"
    _save_progress(result, run_dir, run_name)

    a_rebuttal = _call_stage(
        lambda budget: _ask_rebuttal(
            a_client, cfg.debater_a, debate_prompt, question,
            own=a_open, other=b_open, max_tokens=budget,
        ),
        stage="a_rebuttal",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["a_rebuttal"] = _expand_policy_source_refs(a_rebuttal, items)
    result["status"] = "a_rebuttal_complete"
    _save_progress(result, run_dir, run_name)

    b_rebuttal = _call_stage(
        lambda budget: _ask_rebuttal(
            b_client, cfg.debater_b, debate_prompt, question,
            own=b_open, other=a_open, max_tokens=budget,
        ),
        stage="b_rebuttal",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["b_rebuttal"] = _expand_policy_source_refs(b_rebuttal, items)
    result["status"] = "b_rebuttal_complete"
    _save_progress(result, run_dir, run_name)

    judgment = _call_stage(
        lambda budget: _ask_judge(
        judge_client,
        debate_prompt,
        question,
        cfg.debater_a,
        a_open,
        a_rebuttal,
        cfg.debater_b,
        b_open,
        b_rebuttal,
        max_tokens=budget,
        ),
        stage="judge",
        max_tokens=cfg.max_tokens,
        retries=cfg.stage_retries,
    )
    result["judgment"] = _expand_policy_source_refs(judgment, items)
    result["status"] = "complete"
    _save(result, run_dir, run_name)
    return result


def _ask_debater(client, debater: DebaterConfig, source_prompt: str,
                 question: str, *, max_tokens: int) -> str:
    system = (
        "You are a debate participant analyzing policy/source materials. "
        "Argue your assigned role forcefully but honestly. Cite supplied web "
        "sources as [S1], [S2], etc. Cite Assured Autonomy course concepts as "
        "[AA1], [AA2], etc. Do not invent permissions or legal facts."
    )
    user = (
        f"{source_prompt}\n\n"
        f"Debate question:\n{question}\n\n"
        f"Your assigned role:\n{debater.role}\n\n"
        "Produce your opening argument. Include a section called 'Concept "
        "vectors from Assured Autonomy' with at least five concepts from the "
        "course evidence and how each concept changes the access strategy. "
        "Separate explicit source facts from risk analysis and practical "
        "recommendations."
    )
    return client.complete(system, user, temperature=0.2, max_tokens=max_tokens)


def _ask_rebuttal(client, debater: DebaterConfig, source_prompt: str,
                  question: str, *, own: str, other: str, max_tokens: int) -> str:
    system = (
        "You are a debate participant. Rebut the other side while correcting "
        "your own overclaims if needed. Cite supplied web sources as [S1], [S2], etc. "
        "Cite Assured Autonomy concepts as [AA1], [AA2], etc."
    )
    user = (
        f"{source_prompt}\n\n"
        f"Debate question:\n{question}\n\n"
        f"Your assigned role:\n{debater.role}\n\n"
        f"Your opening argument:\n{own}\n\n"
        f"Other side's opening argument:\n{other}\n\n"
        "Produce a rebuttal with corrections, concessions, remaining disagreements, "
        "and a short note on which Assured Autonomy concepts the other side underused."
    )
    return client.complete(system, user, temperature=0.2, max_tokens=max_tokens)


def _ask_judge(client, source_prompt: str, question: str,
               a: DebaterConfig, a_open: str, a_rebuttal: str,
               b: DebaterConfig, b_open: str, b_rebuttal: str,
               *, max_tokens: int) -> str:
    system = (
        "You are a neutral judge. Synthesize the debate into a final policy memo. "
        "Reward source-grounded claims, penalize overclaims, and distinguish "
        "web-source facts from risk analysis. Cite supplied web sources as [S1], "
        "[S2], etc. Cite Assured Autonomy concepts as [AA1], [AA2], etc. "
        "Explicitly identify which course concepts were most useful."
    )
    user = (
        f"{source_prompt}\n\n"
        f"Debate question:\n{question}\n\n"
        f"Debater A ({a.provider}/{a.model}) role:\n{a.role}\n\n"
        f"A opening:\n{a_open}\n\nA rebuttal:\n{a_rebuttal}\n\n"
        f"Debater B ({b.provider}/{b.model}) role:\n{b.role}\n\n"
        f"B opening:\n{b_open}\n\nB rebuttal:\n{b_rebuttal}\n\n"
        "Produce: 1) winner by issue, 2) Assured Autonomy concept map, "
        "3) final answer, 4) strongest evidence, 5) overclaims to avoid, "
        "6) recommended request strategy, 7) draft email."
    )
    return client.complete(system, user, temperature=0.1, max_tokens=max_tokens)


def _retrieve_aa_concepts(cfg: PolicyDebateConfig, question: str) -> dict:
    index = RagIndex.load(cfg.index_dir)
    retriever = HybridPaperRetriever(
        index,
        corpus_dir=cfg.papers_dir,
        cfg=HybridRetrievalConfig(top_k=cfg.aa_top_k_per_concept),
    )
    concept_queries = [
        (
            "permission_governance",
            "governance accountability authorization permitted use policy gaps responsible AI dataset access",
        ),
        (
            "privacy_minimization",
            "privacy personal data data minimization restricted access sharing downstream use autonomous systems",
        ),
        (
            "misuse_and_dual_use",
            "misuse unintended use adversarial use security data access model training governance",
        ),
        (
            "documentation_traceability",
            "documentation traceability assurance case verification validation data provenance reproducibility",
        ),
        (
            "human_oversight_accountability",
            "human oversight accountability stakeholder responsibility auditability transparency governance",
        ),
        (
            "operational_constraints",
            "operational constraints runtime monitoring assurance boundaries safe use deployment context",
        ),
    ]
    hits: list[tuple[str, Hit]] = []
    seen: set[str] = set()
    for concept, query_text in concept_queries:
        full_query = f"{question}\n{query_text}"
        for hit in retriever.retrieve(full_query, top_k=cfg.aa_top_k_per_concept):
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            hits.append((concept, hit))

    lines: list[str] = []
    items: list[dict] = []
    for i, (concept, hit) in enumerate(hits, start=1):
        handle = f"AA{i}"
        path = cfg.papers_dir / hit.chunk.relpath
        source = _source(hit)
        lines.append(
            f"[{handle}] Concept: {concept}; Source: {source}; "
            f"Module: {hit.chunk.module_name}; Local Path: {path}\n"
            f"{hit.chunk.text[:1000]}"
        )
        items.append({
            "handle": handle,
            "concept": concept,
            "source": source,
            "module": hit.chunk.module_name,
            "path": str(path),
            "text_preview": hit.chunk.text[:400],
        })
    return {
        "prompt_block": "\n\n".join(lines) if lines else "(no Assured Autonomy evidence retrieved)",
        "items": items,
    }


def _source(hit: Hit) -> str:
    parts = Path(hit.chunk.filename).stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return Path(hit.chunk.filename).stem


def _call_stage(call, *, stage: str, max_tokens: int, retries: int) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        budget = max(1200, int(max_tokens * (0.7 ** attempt)))
        try:
            return call(budget)
        except Exception as exc:
            last_exc = exc
            print(f"  ! {stage} failed on attempt {attempt + 1} with {budget} tokens: {exc}")
    raise RuntimeError(f"{stage} failed after {retries + 1} attempts") from last_exc


def _new_run_dir(cfg: PolicyDebateConfig) -> tuple[Path, str]:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name = (
        f"policy_debate_{_safe_name(cfg.debater_a.model)}_vs_"
        f"{_safe_name(cfg.debater_b.model)}_{stamp}"
    )
    run_dir = cfg.out_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, name


def _save_progress(result: dict, run_dir: Path, name: str) -> None:
    json_path = run_dir / f"{name}.partial.json"
    md_path = run_dir / f"{name}.partial.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote progress {json_path}")


def _save(result: dict, run_dir: Path, name: str) -> None:
    json_path = run_dir / f"{name}.json"
    md_path = run_dir / f"{name}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Saved debate directory: {run_dir}")


def _render_markdown(result: dict) -> str:
    lines = [
        "# Policy Debate",
        "",
        "## Question",
        result["question"],
        "",
        "## Sources",
        "```json",
        json.dumps(result["policy_source"], indent=2),
        "```",
        "",
        "## Assured Autonomy Evidence",
        "```json",
        json.dumps(result["assured_autonomy_evidence"], indent=2),
        "```",
        "",
        f"## Debater A: {result['debater_a']['provider']} / {result['debater_a']['model']}",
        f"Role: {result['debater_a']['role']}",
        "",
        result["a_open"],
        "",
        "## Debater B: "
        f"{result['debater_b']['provider']} / {result['debater_b']['model']}",
        f"Role: {result['debater_b']['role']}",
        "",
        result["b_open"],
        "",
        "## A Rebuttal",
        result["a_rebuttal"],
        "",
        "## B Rebuttal",
        result["b_rebuttal"],
        "",
        "## Judge",
        result["judgment"],
    ]
    return "\n".join(lines) + "\n"

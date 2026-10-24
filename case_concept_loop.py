"""Turn an applied policy debate into a second-stage corpus concept debate."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from .concept_debate import ConceptDebateConfig, run_concept_debate
from .llm import get_llm


@dataclass
class CaseConceptLoopConfig:
    case_run: Path
    extractor_provider: str = "openai"
    extractor_model: str = "gpt-4.1"
    provider_a: str = "openai"
    model_a: str = "gpt-4.1"
    provider_b: str = "mistral"
    model_b: str = "mistral-large-latest"
    judge_provider: str = "openai"
    judge_model: str = "gpt-4.1"
    index_dir: Path = Path(".aate_index")
    papers_dir: Path = Path('papers')
    top_k: int = 7
    max_tokens: int = 5000
    extractor_max_tokens: int = 2500
    stage_retries: int = 2
    max_case_chars: int = 70000
    out_dir: Path = Path("concept_runs")


def run_case_concept_loop(cfg: CaseConceptLoopConfig) -> dict:
    case = _load_case_run(cfg.case_run)
    case_digest = _case_digest(case, max_chars=cfg.max_case_chars)
    extractor = get_llm(cfg.extractor_provider, model=cfg.extractor_model)
    concept_question = extractor.complete(
        "Extract corpus-dependent Assured Autonomy questions from an applied case debate.",
        (
            "You are preparing a second-stage debate that must use only the Assured "
            "Autonomy course corpus. Read the applied case/debate below and extract "
            "only the claims whose correctness depends on interpreting course concepts, "
            "not on re-reading the external policy facts.\n\n"
            "Return a single debate question plus bullet points. Include tensions, "
            "weak analogies, and concepts that need better corpus grounding. Do not "
            "answer the case; formulate what the corpus debate must resolve.\n\n"
            f"Applied case/debate digest:\n{case_digest}"
        ),
        temperature=0.1,
        max_tokens=cfg.extractor_max_tokens,
    )

    run_dir = _new_loop_dir(cfg.out_dir)
    bridge = {
        "case_run": str(cfg.case_run),
        "extractor": {"provider": cfg.extractor_provider, "model": cfg.extractor_model},
        "concept_debate": {
            "provider_a": cfg.provider_a,
            "model_a": cfg.model_a,
            "provider_b": cfg.provider_b,
            "model_b": cfg.model_b,
            "judge_provider": cfg.judge_provider,
            "judge_model": cfg.judge_model,
        },
        "generated_concept_question": concept_question,
    }
    _save_bridge(bridge, run_dir)

    concept_cfg = ConceptDebateConfig(
        index_dir=cfg.index_dir,
        papers_dir=cfg.papers_dir,
        provider_a=cfg.provider_a,
        model_a=cfg.model_a,
        provider_b=cfg.provider_b,
        model_b=cfg.model_b,
        judge_provider=cfg.judge_provider,
        judge_model=cfg.judge_model,
        top_k=cfg.top_k,
        max_tokens=cfg.max_tokens,
        stage_retries=cfg.stage_retries,
        out_dir=run_dir,
    )
    concept_result = run_concept_debate(concept_cfg, concept_question)
    bridge["concept_result_status"] = concept_result.get("status")
    bridge["concept_result_question"] = concept_result.get("question")
    _save_bridge(bridge, run_dir)
    print(f"Saved case-to-concept loop directory: {run_dir}")
    return bridge


def _load_case_run(path: Path) -> dict:
    if path.is_dir():
        jsons = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        jsons = [p for p in jsons if not p.name.endswith(".partial.json")]
        if not jsons:
            raise FileNotFoundError(f"No completed JSON run found in {path}")
        path = jsons[0]
    return json.loads(path.read_text(encoding="utf-8"))


def _case_digest(case: dict, *, max_chars: int) -> str:
    parts = []
    for key in (
        "question",
        "policy_source",
        "assured_autonomy_evidence",
        "a_open",
        "b_open",
        "a_rebuttal",
        "b_rebuttal",
        "judgment",
        "answer",
        "critique",
    ):
        if key not in case:
            continue
        value = case[key]
        text = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
        parts.append(f"## {key}\n{text}")
    digest = "\n\n".join(parts)
    if len(digest) <= max_chars:
        return digest
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return digest[:head] + "\n\n[...case digest truncated...]\n\n" + digest[-tail:]


def _new_loop_dir(out_dir: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = out_dir / f"case_concept_loop_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_bridge(result: dict, run_dir: Path) -> None:
    json_path = run_dir / "case_concept_loop.json"
    md_path = run_dir / "case_concept_loop.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join([
            "# Case To Concept Loop",
            "",
            f"Case run: `{result['case_run']}`",
            "",
            "## Generated Corpus Debate Question",
            result["generated_concept_question"],
            "",
        ]),
        encoding="utf-8",
    )

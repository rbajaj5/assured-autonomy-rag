"""Upgrade selected course readings into code-annotation-grade cards."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any

from .llm import get_llm
from .pdf_text import extract_pdf_pages


TARGET_PAPERS = [
    r"papers\papers\M07_TestEvaluation\Tuncali-2018-SimulationBasedAdversarialTestGen.pdf",
    r"papers\papers\M05_DataSetShift\Sinha-2022-SystemLevelOODRobotics.pdf",
    r"papers\papers\M12_RuntimeAssurance\Desai-2019-SOTER.pdf",
    r"papers\papers\M13_SafeRL\Krasowski-2023-ProvablySafeRLSurvey.pdf",
    r"papers\papers\M06_VerificationValidation\Seshia-2015-FormalMethodsSemiAutonomousDriving.pdf",
]

CHUNK_FIELDS = [
    "page_range",
    "actual_argument",
    "method_or_evidence",
    "assumptions",
    "limits",
    "assurance_concepts",
    "code_annotation_tags",
    "tests_or_metrics_implied",
    "overclaim_warnings",
    "citation_anchors",
]

PAPER_FIELDS = [
    "paper_identity",
    "actual_contribution",
    "method_or_framework",
    "assurance_concepts",
    "code_annotation_tags",
    "codebase_mapping",
    "tests_or_metrics_implied",
    "assumptions_and_limits",
    "overclaim_warnings",
    "page_citation_anchors",
    "study_questions",
]


PROJECT_CONTEXT = """
We are building an Assured Autonomy RAG for a course project on autonomous-driving
assurance. The candidate code stack includes:
- Scenic: parameterized scenario generation and scenario-family X/Y definitions.
- CARLA ScenarioRunner: execution, criteria, logs, infractions, metrics, OpenSCENARIO.
- SP_RL: SafetyPotential-shaped safe-RL longitudinal-control baseline in CARLA.
- safe_control: CBF/reachability/safety-filter control baseline.
- REAP/NeuroNCAP: adversarial perception and closed-loop safety stress tests.
- ROSMonitoring/Ogma: runtime monitoring and formal/specification support.

The central project question is sparse-test trust transfer: if a system succeeds
on scenario family X, when does that evidence justify limited trust on scenario
family Y? The upgraded cards must support code annotations that tie specific
files/functions to course concepts, citations, assumptions, tests, metrics, and
overclaim warnings.
""".strip()


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


def _valid(card: dict[str, Any], fields: list[str]) -> tuple[bool, str]:
    missing = [f for f in fields if not card.get(f)]
    if missing:
        return False, "missing fields: " + ", ".join(missing)
    return True, ""


def _safe_stem(path: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", path).strip("-")


def _chunk_pages(pdf_path: Path, *, chunk_chars: int, max_pages: int) -> list[dict[str, Any]]:
    pages = extract_pdf_pages(pdf_path, max_pages=max_pages)
    chunks = []
    current: list[str] = []
    start_page = 1
    current_len = 0
    for i, text in enumerate(pages, start=1):
        text = re.sub(r"[ \t]+", " ", text or "").strip()
        if not text:
            continue
        labeled = f"[Page {i}]\n{text}"
        if current and current_len + len(labeled) > chunk_chars:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_start": start_page,
                "page_end": i - 1,
                "text": "\n\n".join(current),
            })
            current = []
            start_page = i
            current_len = 0
        current.append(labeled)
        current_len += len(labeled)
    if current:
        last_pages = re.findall(r"\[Page (\d+)\]", "\n\n".join(current))
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "page_start": start_page,
            "page_end": int(last_pages[-1]) if last_pages else start_page,
            "text": "\n\n".join(current),
        })
    return chunks


def _paper_meta(pdf: Path) -> dict[str, Any]:
    rel = str(pdf).replace("\\", "/")
    module = next((part for part in pdf.parts if re.match(r"M\d{2}|M01-02", part)), "")
    module_id = module.split("_", 1)[0] if module else ""
    return {
        "path": str(pdf),
        "short_path": rel.split("papers/papers/", 1)[-1] if "papers/papers/" in rel else rel,
        "filename": pdf.name,
        "module_dir": module,
        "module_id": module_id,
    }


def _build_chunk_card(llm: Any, paper: dict[str, Any], chunk: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    system = (
        "You create code-annotation-grade reading cards for Assured Autonomy. "
        "Return valid JSON only. Be specific about what the paper actually says, "
        "what it does not prove, and how it maps to code tags."
    )
    user = f"""
Paper:
{json.dumps(paper, indent=2)}

Project context:
{PROJECT_CONTEXT}

Chunk pages {chunk['page_start']}-{chunk['page_end']}:
{chunk['text']}

Return JSON with exactly these fields:
{json.dumps(CHUNK_FIELDS)}

Field guidance:
- actual_argument: 3-8 precise claims made in this chunk.
- method_or_evidence: concrete methods, models, experiments, theorem/proof style, evaluation setup, or survey taxonomy evidence.
- assumptions: assumptions that code annotations must remember.
- limits: limits and failure cases.
- assurance_concepts: concepts such as scenario generation, dataset shift, runtime monitor, reward shaping, formal model, test oracle, safety envelope.
- code_annotation_tags: list of objects with tag, definition, maps_to_possible_code, citation_use.
- tests_or_metrics_implied: concrete tests/metrics a codebase should implement.
- overclaim_warnings: claims that would be too strong.
- citation_anchors: short page-specific anchors, each with page and citation_text paraphrase.
""".strip()
    raw = llm.complete(system, user, temperature=0.1, max_tokens=max_tokens)
    card = _json_from_text(raw)
    ok, err = _valid(card, CHUNK_FIELDS)
    card.update({
        "short_path": paper["short_path"],
        "chunk_id": chunk["chunk_id"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "raw_model_output": raw,
        "status": "complete" if ok else "failed",
    })
    if err:
        card["validation_error"] = err
    return card


def _build_paper_card(llm: Any, paper: dict[str, Any], chunk_cards: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    system = (
        "You synthesize code-annotation-grade cards from chunk cards. Return valid JSON only. "
        "The output will drive sidecar comments for Python repositories, so it must be concrete, "
        "citation-aware, and careful about limits."
    )
    compact_chunks = [
        {k: c.get(k) for k in CHUNK_FIELDS + ["page_start", "page_end", "chunk_id"]}
        for c in chunk_cards
        if c.get("status") == "complete"
    ]
    user = f"""
Paper:
{json.dumps(paper, indent=2)}

Project context:
{PROJECT_CONTEXT}

Chunk cards:
{json.dumps(compact_chunks, indent=2, ensure_ascii=False)[:90000]}

Return JSON with exactly these fields:
{json.dumps(PAPER_FIELDS)}

Field guidance:
- paper_identity: title/authors/year if available plus local path.
- actual_contribution: what this paper contributes, not generic field background.
- method_or_framework: specific method/taxonomy/framework/evidence used.
- assurance_concepts: high-value concepts for the Assured Autonomy concept graph.
- code_annotation_tags: list of tags with tag, definition, cite_as, applies_to_code_files, relevant_metrics, limit_warning.
- codebase_mapping: map to Scenic, ScenarioRunner, SP_RL, safe_control, REAP/NeuroNCAP, ROSMonitoring/Ogma when applicable.
- tests_or_metrics_implied: exact tests/metrics to add to code or experiment harnesses.
- assumptions_and_limits: conditions under which the paper's idea is valid.
- overclaim_warnings: concise warnings suitable for code sidecar annotations.
- page_citation_anchors: page-specific anchors suitable for citations in code tags.
- study_questions: questions a student should be able to answer after reading.
""".strip()
    raw = llm.complete(system, user, temperature=0.1, max_tokens=max_tokens)
    card = _json_from_text(raw)
    ok, err = _valid(card, PAPER_FIELDS)
    card.update({
        "short_path": paper["short_path"],
        "filename": paper["filename"],
        "module_id": paper["module_id"],
        "raw_model_output": raw,
        "status": "complete" if ok else "failed",
        "chunk_count": len(chunk_cards),
    })
    if err:
        card["validation_error"] = err
    return card


def _render_md(card: dict[str, Any]) -> str:
    lines = [f"# {card.get('filename')}", ""]
    for field in PAPER_FIELDS:
        lines.append(f"## {field.replace('_', ' ').title()}")
        value = card.get(field)
        if isinstance(value, (list, dict)):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append(str(value or ""))
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.out_dir) / f"code_reading_upgrade_{args.model.replace('.', '-')}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    chunk_dir = run_dir / "chunk_cards"
    paper_dir = run_dir / "paper_cards"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    llm = get_llm(args.provider, model=args.model)
    selected = [Path(p) for p in (args.paper or TARGET_PAPERS)]
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "provider": args.provider,
        "model": args.model,
        "papers": [],
        "project_context": PROJECT_CONTEXT,
    }

    for idx, pdf in enumerate(selected, start=1):
        if not pdf.exists():
            print(f"[{idx}/{len(selected)}] missing {pdf}")
            continue
        paper = _paper_meta(pdf)
        print(f"[{idx}/{len(selected)}] upgrade {paper['short_path']}")
        chunks = _chunk_pages(pdf, chunk_chars=args.chunk_chars, max_pages=args.max_pages)
        chunk_cards = []
        for chunk in chunks:
            card_path = chunk_dir / f"{_safe_stem(paper['short_path'])}__chunk-{chunk['chunk_id']:03d}.json"
            card = None
            for attempt in range(1, args.retries + 2):
                card = _build_chunk_card(llm, paper, chunk, max_tokens=args.chunk_max_tokens)
                if card.get("status") == "complete":
                    break
                print(f"  ! chunk {chunk['chunk_id']} invalid attempt {attempt}: {card.get('validation_error')}")
            card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
            chunk_cards.append(card)
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
        paper_card = _build_paper_card(llm, paper, chunk_cards, max_tokens=args.paper_max_tokens)
        paper_json = paper_dir / f"{_safe_stem(paper['short_path'])}.json"
        paper_json.write_text(json.dumps(paper_card, indent=2, ensure_ascii=False), encoding="utf-8")
        (paper_dir / f"{_safe_stem(paper['short_path'])}.md").write_text(_render_md(paper_card), encoding="utf-8")
        manifest["papers"].append({
            "short_path": paper["short_path"],
            "chunks": len(chunks),
            "paper_card": str(paper_json),
            "status": paper_card.get("status"),
        })
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved code reading upgrade directory: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--paper", action="append")
    parser.add_argument("--out-dir", default="code_reading_upgrade_runs")
    parser.add_argument("--chunk-chars", type=int, default=16000)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--chunk-max-tokens", type=int, default=2600)
    parser.add_argument("--paper-max-tokens", type=int, default=7000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

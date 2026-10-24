"""Comprehensive chunk-by-chunk semantic sweep for course and external readings."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AA_MODULES, module_id_from_dirname
from .llm import get_llm
from .pdf_text import extract_pdf_pages


@dataclass
class FullSweepConfig:
    papers_dirs: list[Path]
    provider: str = "openai"
    model: str = "gpt-5.5"
    out_dir: Path = Path("full_sweep_runs")
    resume_dir: Path | None = None
    case_file: Path | None = None
    case_text: str | None = None
    modules: list[str] | None = None
    limit: int | None = None
    chunk_chars: int = 18000
    chunk_max_tokens: int = 2200
    paper_max_tokens: int = 5200
    synthesis_max_tokens: int = 9000
    retries: int = 2
    sleep_seconds: float = 0.0
    force: bool = False
    synthesize_only: bool = False
    allow_failed_chunks: bool = False


DEFAULT_CASE = (
    "Build a TechTrack upgrade for German traffic-sign perception. Study neural "
    "network capacity allocation in YOLO/CNN models, including input resolution, "
    "anchor geometry, small-object heads, robust objectives, calibration, hard "
    "negative mining, spectral diagnostics, shift/adversarial tests, and bounded "
    "verification. The goal is an Assured Autonomy experiment, not a deployment claim."
)


CHUNK_FIELDS = (
    "chunk_claims",
    "methods_or_evidence",
    "assumptions_and_limits",
    "specific_concepts",
    "capacity_allocation_hooks",
    "techtrack_upgrade_hooks",
    "assurance_tests_or_metrics",
    "overclaim_warnings",
    "page_anchors",
)


PAPER_FIELDS = (
    "paper_central_claim",
    "method_or_framework",
    "specific_concepts",
    "capacity_allocation_relevance",
    "techtrack_upgrade_relevance",
    "assurance_tests_or_metrics",
    "assumptions_limits",
    "misuse_or_overclaim_warnings",
    "citation_snippets",
)


def run_full_sweep(cfg: FullSweepConfig) -> dict:
    run_dir = cfg.resume_dir or _new_run_dir(cfg.out_dir, cfg.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = run_dir / "chunk_cards"
    paper_dir = run_dir / "paper_cards"
    chunks_dir.mkdir(exist_ok=True)
    paper_dir.mkdir(exist_ok=True)

    papers = _discover_papers(cfg.papers_dirs, cfg.modules)
    if cfg.limit:
        papers = papers[: cfg.limit]
    case_context = _case_context(cfg)
    manifest = {
        "run_dir": str(run_dir),
        "papers_dirs": [str(p) for p in cfg.papers_dirs],
        "provider": cfg.provider,
        "model": cfg.model,
        "chunk_chars": cfg.chunk_chars,
        "papers_total": len(papers),
        "papers": [],
    }
    (run_dir / "case_context.txt").write_text(case_context, encoding="utf-8")
    llm = get_llm(cfg.provider, model=cfg.model)

    if not cfg.synthesize_only:
        for i, paper in enumerate(papers, start=1):
            print(f"[{i}/{len(papers)}] digest {paper['short_path']}")
            paper_chunks = _chunk_pdf(paper["path"], cfg.chunk_chars)
            chunk_cards = []
            for chunk in paper_chunks:
                card_path = chunks_dir / f"{_safe_stem(paper['short_path'])}__chunk-{chunk['chunk_id']:03d}.json"
                if card_path.exists() and not cfg.force:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                    if card.get("status") != "complete" or card.get("validation_error"):
                        print(f"  ! rebuilding failed existing chunk {chunk['chunk_id']}")
                        card = _build_chunk_card_with_retries(llm, paper, chunk, case_context, cfg)
                        if card.get("status") == "failed" and not cfg.allow_failed_chunks:
                            reason = card.get("validation_error", "unknown validation error")
                            raise RuntimeError(
                                f"Chunk card failed for {paper['short_path']} chunk {chunk['chunk_id']}: {reason}. "
                                "Resume after increasing token budgets/retries, or pass --allow-failed-chunks."
                            )
                        card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
                else:
                    card = _build_chunk_card_with_retries(llm, paper, chunk, case_context, cfg)
                    if card.get("status") == "failed" and not cfg.allow_failed_chunks:
                        reason = card.get("validation_error", "unknown validation error")
                        raise RuntimeError(
                            f"Chunk card failed for {paper['short_path']} chunk {chunk['chunk_id']}: {reason}. "
                            "Resume after increasing token budgets/retries, or pass --allow-failed-chunks."
                        )
                    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
                    if cfg.sleep_seconds:
                        time.sleep(cfg.sleep_seconds)
                chunk_cards.append(card)
            usable_chunk_cards = [
                card for card in chunk_cards
                if card.get("status") == "complete" and not card.get("validation_error")
            ]
            if len(usable_chunk_cards) != len(chunk_cards):
                print(f"  ! excluding {len(chunk_cards) - len(usable_chunk_cards)} failed chunks from paper synthesis")
            paper_card_path = paper_dir / f"{_safe_stem(paper['short_path'])}.json"
            if paper_card_path.exists() and not cfg.force:
                paper_card = json.loads(paper_card_path.read_text(encoding="utf-8"))
            else:
                paper_card = _build_paper_card_with_retries(llm, paper, usable_chunk_cards, case_context, cfg)
                paper_card_path.write_text(json.dumps(paper_card, indent=2), encoding="utf-8")
                (paper_dir / f"{_safe_stem(paper['short_path'])}.md").write_text(
                    _render_paper_card_md(paper_card),
                    encoding="utf-8",
                )
            manifest["papers"].append({
                "short_path": paper["short_path"],
                "chunks": len(paper_chunks),
                "paper_card": str(paper_card_path),
            })
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paper_cards = _load_cards(paper_dir)
    synthesis = _synthesize_full_network(llm, paper_cards, case_context, cfg)
    result = {
        "manifest": manifest,
        "paper_card_count": len(paper_cards),
        "synthesis": synthesis,
    }
    (run_dir / "full_reading_synthesis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "full_reading_synthesis.md").write_text(synthesis, encoding="utf-8")
    print(f"Saved full reading sweep directory: {run_dir}")
    return result


def _discover_papers(roots: list[Path], modules: list[str] | None) -> list[dict]:
    wanted = set(modules or [])
    rows = []
    seen = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pdf")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            module_dir = next((p for p in path.parents if module_id_from_dirname(p.name)), None)
            module_id = module_id_from_dirname(module_dir.name) if module_dir else "EXT"
            if wanted and module_id not in wanted and (module_dir is None or module_dir.name not in wanted):
                continue
            module_name = AA_MODULES.get(module_id or "", module_dir.name if module_dir else "External Reading")
            rel_root = root
            try:
                rel = path.relative_to(rel_root)
            except ValueError:
                rel = path.name
            rows.append({
                "path": path,
                "short_path": str(rel).replace("\\", "/"),
                "module_id": module_id,
                "module_name": module_name,
                "filename": path.name,
            })
    return rows


def _chunk_pdf(path: Path, chunk_chars: int) -> list[dict]:
    pages = extract_pdf_pages(path, max_pages=500)
    chunks = []
    current = []
    start_page = 1
    current_len = 0
    for page_no, page_text in enumerate(pages, start=1):
        text = re.sub(r"[ \t]+", " ", page_text or "").strip()
        if not text:
            continue
        labeled = f"[Page {page_no}]\n{text}"
        if current and current_len + len(labeled) > chunk_chars:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_start": start_page,
                "page_end": page_no - 1,
                "text": "\n\n".join(current),
            })
            current = []
            start_page = page_no
            current_len = 0
        current.append(labeled)
        current_len += len(labeled)
    if current:
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "page_start": start_page,
            "page_end": chunks[-1]["page_end"] + 1 if chunks else len(pages),
            "text": "\n\n".join(current),
        })
        chunks[-1]["page_end"] = max(chunks[-1]["page_start"], _last_page_in_text(chunks[-1]["text"]))
    return chunks


def _last_page_in_text(text: str) -> int:
    matches = re.findall(r"\[Page (\d+)\]", text)
    return int(matches[-1]) if matches else 1


def _build_chunk_card_with_retries(llm, paper: dict, chunk: dict, case_context: str, cfg: FullSweepConfig) -> dict:
    last = {}
    for attempt in range(1, cfg.retries + 2):
        card = _build_chunk_card(llm, paper, chunk, case_context, cfg, attempt)
        if _valid(card, CHUNK_FIELDS):
            card["status"] = "complete"
            return card
        raw_len = len(str(card.get("raw_model_output", "") or ""))
        print(
            f"  ! chunk {chunk['chunk_id']} invalid attempt {attempt}: "
            f"{card.get('validation_error', 'invalid')} (raw chars={raw_len})"
        )
        last = card
    last["status"] = "failed"
    return last


def _build_chunk_card(llm, paper: dict, chunk: dict, case_context: str, cfg: FullSweepConfig, attempt: int) -> dict:
    system = (
        "You are creating a study-grade semantic card for one chunk of an academic PDF. "
        "Extract specific arguments, methods, assumptions, and experiment implications. "
        "Return valid JSON only."
    )
    user = (
        f"Paper: {paper['short_path']}\n"
        f"Module: {paper['module_id']} {paper['module_name']}\n"
        f"Chunk: {chunk['chunk_id']} pages {chunk['page_start']}-{chunk['page_end']}\n\n"
        f"Project context:\n{case_context}\n\n"
        "Return one complete JSON object and no markdown. Every listed key is required. "
        "If a key has weak evidence, say so explicitly rather than omitting it.\n\n"
        "Required keys: short_path, module_id, module_name, chunk_id, page_start, page_end, "
        "chunk_claims, methods_or_evidence, assumptions_and_limits, specific_concepts, "
        "capacity_allocation_hooks, techtrack_upgrade_hooks, assurance_tests_or_metrics, "
        "overclaim_warnings, page_anchors.\n\n"
        "Make page_anchors compact paraphrased anchors with page numbers. No long quotes. "
        "Focus especially on how this chunk affects neural capacity allocation, perception assurance, "
        "testing, runtime assurance, dataset shift, adversarial robustness, or verification.\n\n"
        f"Attempt: {attempt}\n\n"
        f"Chunk text:\n{chunk['text']}"
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=cfg.chunk_max_tokens * (attempt + 1))
    obj = _parse_json(raw)
    if not raw.strip():
        obj["validation_error"] = "empty model output"
    obj.setdefault("short_path", paper["short_path"])
    obj.setdefault("module_id", paper["module_id"])
    obj.setdefault("module_name", paper["module_name"])
    obj.setdefault("chunk_id", chunk["chunk_id"])
    obj.setdefault("page_start", chunk["page_start"])
    obj.setdefault("page_end", chunk["page_end"])
    obj["raw_model_output"] = obj.get("raw_model_output", raw)
    return obj


def _build_paper_card_with_retries(llm, paper: dict, chunk_cards: list[dict], case_context: str, cfg: FullSweepConfig) -> dict:
    last = {}
    for attempt in range(1, cfg.retries + 2):
        card = _build_paper_card(llm, paper, chunk_cards, case_context, cfg, attempt)
        if _valid(card, PAPER_FIELDS):
            card["status"] = "complete"
            return card
        print(f"  ! paper card invalid attempt {attempt}: {card.get('validation_error', 'invalid')}")
        last = card
    last["status"] = "failed"
    return last


def _build_paper_card(llm, paper: dict, chunk_cards: list[dict], case_context: str, cfg: FullSweepConfig, attempt: int) -> dict:
    compact = []
    for card in chunk_cards:
        compact.append({
            "chunk_id": card.get("chunk_id"),
            "pages": f"{card.get('page_start')}-{card.get('page_end')}",
            "claims": card.get("chunk_claims"),
            "concepts": card.get("specific_concepts"),
            "capacity": card.get("capacity_allocation_hooks"),
            "techtrack": card.get("techtrack_upgrade_hooks"),
            "tests": card.get("assurance_tests_or_metrics"),
            "warnings": card.get("overclaim_warnings"),
            "anchors": card.get("page_anchors"),
        })
    system = (
        "You are synthesizing chunk cards into a study-grade paper card. "
        "Do not make a generic abstract. Preserve the paper's actual argument and how it supports the project. "
        "Return valid JSON only."
    )
    user = (
        f"Paper: {paper['short_path']}\n"
        f"Module: {paper['module_id']} {paper['module_name']}\n\n"
        f"Project context:\n{case_context}\n\n"
        "Return one complete JSON object and no markdown. Every listed key is required. "
        "If a key has weak evidence, say so explicitly rather than omitting it.\n\n"
        "Required keys: short_path, module_id, module_name, filename, coverage_chunks, "
        "paper_central_claim, method_or_framework, specific_concepts, capacity_allocation_relevance, "
        "techtrack_upgrade_relevance, assurance_tests_or_metrics, assumptions_limits, "
        "misuse_or_overclaim_warnings, citation_snippets, presentation_questions.\n\n"
        "citation_snippets must cite compact paths and page numbers. Explain how the paper should or should not "
        "shape the TechTrack German-sign upgrade.\n\n"
        f"Attempt: {attempt}\n\n"
        f"Chunk cards:\n{json.dumps(compact, indent=2)}"
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=cfg.paper_max_tokens * (attempt + 1))
    obj = _parse_json(raw)
    if not raw.strip():
        obj["validation_error"] = "empty model output"
    obj.setdefault("short_path", paper["short_path"])
    obj.setdefault("module_id", paper["module_id"])
    obj.setdefault("module_name", paper["module_name"])
    obj.setdefault("filename", paper["filename"])
    obj.setdefault("coverage_chunks", len(chunk_cards))
    obj["raw_model_output"] = obj.get("raw_model_output", raw)
    return obj


def _synthesize_full_network(llm, cards: list[dict], case_context: str, cfg: FullSweepConfig) -> str:
    compact = []
    for card in cards:
        compact.append({
            "short_path": card.get("short_path"),
            "module": f"{card.get('module_id')} {card.get('module_name')}",
            "claim": card.get("paper_central_claim"),
            "capacity": card.get("capacity_allocation_relevance"),
            "techtrack": card.get("techtrack_upgrade_relevance"),
            "tests": card.get("assurance_tests_or_metrics"),
            "warnings": card.get("misuse_or_overclaim_warnings"),
        })
    system = (
        "You are building a comprehensive semantic network for an Assured Autonomy project. "
        "Use the full paper cards substantively and design the TechTrack German-sign upgrade."
    )
    user = (
        f"Project context:\n{case_context}\n\n"
        f"Paper cards:\n{json.dumps(compact, indent=2)}\n\n"
        "Produce a comprehensive synthesis with sections: "
        "1. concept network, 2. capacity allocation theory for YOLO/CNN perception, "
        "3. TechTrack upgrade architecture, 4. model competitors, 5. metrics and tests, "
        "6. Waymo/safety-case comparison criteria, 7. evidence gaps requiring deep PDF search, "
        "8. minimum viable course experiment, 9. workshop-paper hypothesis, 10. overclaim controls. "
        "Cite compact paper paths."
    )
    for attempt in range(1, 4):
        out = llm.complete(system, user, temperature=0.1, max_tokens=cfg.synthesis_max_tokens * attempt)
        if out.strip():
            return out
    raise RuntimeError("Full reading synthesis failed with empty output.")


def _valid(obj: dict, fields: tuple[str, ...]) -> bool:
    missing = []
    for field in fields:
        value = obj.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)
    if missing:
        obj["validation_error"] = "missing or empty fields: " + ", ".join(missing)
        return False
    return True


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {"status": "parse_failed", "raw_model_output": raw, "validation_error": "not valid JSON"}


def _load_cards(path: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]


def _render_paper_card_md(card: dict) -> str:
    return "\n".join([
        f"# {card.get('short_path')}",
        "",
        f"Module: {card.get('module_id')} {card.get('module_name')}",
        "",
        "## Central Claim",
        str(card.get("paper_central_claim", "")),
        "",
        "## Capacity Allocation Relevance",
        str(card.get("capacity_allocation_relevance", "")),
        "",
        "## TechTrack Upgrade Relevance",
        str(card.get("techtrack_upgrade_relevance", "")),
        "",
        "## Tests And Metrics",
        json.dumps(card.get("assurance_tests_or_metrics", []), indent=2),
        "",
        "## Overclaim Warnings",
        json.dumps(card.get("misuse_or_overclaim_warnings", []), indent=2),
        "",
    ])


def _case_context(cfg: FullSweepConfig) -> str:
    if cfg.case_file:
        return cfg.case_file.read_text(encoding="utf-8")
    if cfg.case_text:
        return cfg.case_text
    return DEFAULT_CASE


def _new_run_dir(base: Path, model: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe = _safe_stem(model)
    return base / f"full_reading_sweep_{safe}_{stamp}"


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "item"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", action="append", default=[],
                        help="Directory to scan. Repeat for course and external paper roots.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--out-dir", default="full_sweep_runs")
    parser.add_argument("--resume-dir")
    parser.add_argument("--case-file")
    parser.add_argument("--case-text")
    parser.add_argument("--modules", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-chars", type=int, default=18000)
    parser.add_argument("--chunk-max-tokens", type=int, default=2200)
    parser.add_argument("--paper-max-tokens", type=int, default=5200)
    parser.add_argument("--synthesis-max-tokens", type=int, default=9000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--synthesize-only", action="store_true")
    parser.add_argument("--allow-failed-chunks", action="store_true")
    args = parser.parse_args(argv)
    roots = [Path(p) for p in args.papers_dir] or [
        Path('papers/papers'),
        Path('papers/external'),
    ]
    run_full_sweep(FullSweepConfig(
        papers_dirs=roots,
        provider=args.provider,
        model=args.model,
        out_dir=Path(args.out_dir),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        case_file=Path(args.case_file) if args.case_file else None,
        case_text=args.case_text,
        modules=args.modules,
        limit=args.limit,
        chunk_chars=args.chunk_chars,
        chunk_max_tokens=args.chunk_max_tokens,
        paper_max_tokens=args.paper_max_tokens,
        synthesis_max_tokens=args.synthesis_max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
        synthesize_only=args.synthesize_only,
        allow_failed_chunks=args.allow_failed_chunks,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

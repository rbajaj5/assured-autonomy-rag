"""Build reusable semantic cards for the Assured Autonomy reading corpus."""
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


DPRK_CASE = (
    "DPRK-linked remote IT worker identity-masking schemes: adversaries may use "
    "false/stolen identities, proxy interviewers, laptop relay setups, unauthorized "
    "remote access, payment/account inconsistencies, and deceptive work arrangements. "
    "Organizations must manage insider-threat, data-theft, sanctions, and disruption "
    "risk while avoiding nationality profiling and protecting legitimate Japanese, "
    "Korean, Zainichi Korean, immigrant, multilingual, remote, or accented workers."
)


@dataclass
class SemanticSweepConfig:
    papers_dir: Path
    provider: str = "openai"
    model: str = "gpt-5.5"
    max_doc_chars: int = 45000
    card_max_tokens: int = 2400
    synthesis_max_tokens: int = 7000
    sleep_seconds: float = 0.0
    out_dir: Path = Path("semantic_runs")
    modules: list[str] | None = None
    limit: int | None = None
    force: bool = False
    synthesize_only: bool = False
    case_context: str = DPRK_CASE
    case_name: str = "dprk_protocol"
    card_retries: int = 2
    allow_failed_cards: bool = False


def run_semantic_sweep(cfg: SemanticSweepConfig) -> dict:
    run_dir = _new_or_latest_run_dir(cfg)
    cards_dir = run_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    papers = _discover_papers(cfg.papers_dir, cfg.modules)
    if cfg.limit:
        papers = papers[: cfg.limit]
    manifest = {
        "run_dir": str(run_dir),
        "papers_dir": str(cfg.papers_dir),
        "provider": cfg.provider,
        "model": cfg.model,
        "max_doc_chars": cfg.max_doc_chars,
        "card_max_tokens": cfg.card_max_tokens,
        "synthesis_max_tokens": cfg.synthesis_max_tokens,
        "papers_total": len(papers),
        "cards": [],
    }
    manifest_path = run_dir / "manifest.json"

    llm = get_llm(cfg.provider, model=cfg.model)
    if not cfg.synthesize_only:
        for i, paper in enumerate(papers, start=1):
            card_path = cards_dir / f"{_safe_stem(paper['relpath'])}.json"
            if card_path.exists() and not cfg.force:
                print(f"[{i}/{len(papers)}] skip existing {paper['short_path']}")
                card = json.loads(card_path.read_text(encoding="utf-8"))
            else:
                print(f"[{i}/{len(papers)}] analyze {paper['short_path']}")
                card = _build_card_with_retries(llm, paper, cfg)
                if card.get("status") == "failed" and not cfg.allow_failed_cards:
                    reason = card.get("validation_error", "unknown validation error")
                    raise RuntimeError(
                        f"Semantic card failed for {paper['short_path']}: {reason}. "
                        "Use --allow-failed-cards to continue anyway."
                    )
                card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
                (cards_dir / f"{_safe_stem(paper['relpath'])}.md").write_text(
                    _render_card_md(card),
                    encoding="utf-8",
                )
                if cfg.sleep_seconds:
                    time.sleep(cfg.sleep_seconds)
            manifest["cards"].append({
                "short_path": paper["short_path"],
                "card_path": str(card_path),
                "status": card.get("status", "complete"),
            })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cards = _load_cards(cards_dir)
    synthesis = _synthesize_protocol_with_retries(llm, cards, cfg)
    result = {
        "manifest": manifest,
        "card_count": len(cards),
        "synthesis": synthesis,
    }
    out_stem = f"semantic_network_{_safe_name(cfg.case_name)}"
    (run_dir / f"{out_stem}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (run_dir / f"{out_stem}.md").write_text(
        synthesis,
        encoding="utf-8",
    )
    print(f"Saved semantic sweep directory: {run_dir}")
    return result


def _new_or_latest_run_dir(cfg: SemanticSweepConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.synthesize_only:
        candidates = sorted(
            cfg.out_dir.glob("semantic_sweep_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No semantic_sweep_* directory found for --synthesize-only")
        return candidates[0]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = cfg.out_dir / f"semantic_sweep_{_safe_name(cfg.model)}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _discover_papers(papers_dir: Path, modules: list[str] | None) -> list[dict]:
    wanted = set(modules or [])
    rows = []
    for path in sorted(papers_dir.rglob("*.pdf")):
        module_dir = next((p for p in path.parents if module_id_from_dirname(p.name)), None)
        if not module_dir:
            continue
        module_id = module_id_from_dirname(module_dir.name)
        if wanted and module_id not in wanted and module_dir.name not in wanted:
            continue
        relpath = path.relative_to(papers_dir)
        rows.append({
            "path": path,
            "relpath": str(relpath),
            "short_path": str(relpath).replace("\\", "/"),
            "module_id": module_id,
            "module_name": AA_MODULES.get(module_id or "", module_dir.name),
            "filename": path.name,
        })
    return rows


def _paper_text(path: Path, max_chars: int) -> tuple[str, bool]:
    pages = extract_pdf_pages(path, max_pages=500)
    text = "\n\n".join(
        f"[Page {i}]\n{page}"
        for i, page in enumerate(pages, start=1)
        if page.strip()
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n\n[...document truncated for semantic card...]\n\n" + text[-tail:], True


REQUIRED_CARD_FIELDS = (
    "paper_central_claim",
    "method_or_framework",
    "specific_concepts",
    "assumptions_limits",
    "case_mapping",
    "protocol_controls",
    "misuse_or_overclaim_warnings",
    "citation_snippets",
)


def _build_card_with_retries(llm, paper: dict, cfg: SemanticSweepConfig) -> dict:
    last: dict | None = None
    for attempt in range(cfg.card_retries + 1):
        card = _build_card(llm, paper, cfg, attempt=attempt + 1)
        if _is_valid_card(card):
            card["status"] = "complete"
            return card
        last = card
        reason = card.get("validation_error", "invalid semantic card")
        print(f"  ! card invalid on attempt {attempt + 1}: {reason}")
    assert last is not None
    last["status"] = "failed"
    return last


def _build_card(llm, paper: dict, cfg: SemanticSweepConfig, *, attempt: int = 1) -> dict:
    text, truncated = _paper_text(paper["path"], cfg.max_doc_chars)
    system = (
        "You are building a semantic network card for an Assured Autonomy course reading. "
        "Do not summarize generically. Extract the paper's specific argument and how that "
        "argument can be used in the target project. Return valid JSON only."
    )
    user = (
        f"Reading metadata:\n"
        f"- Module: {paper['module_id']} {paper['module_name']}\n"
        f"- Short path: {paper['short_path']}\n"
        f"- Filename: {paper['filename']}\n"
        f"- Text truncated: {truncated}\n\n"
        f"Target case:\n{DPRK_CASE}\n\n"
        f"Target project for this run:\n{cfg.case_context}\n\n"
        "Produce a JSON object only, with these keys:\n"
        "short_path, module_id, module_name, paper_central_claim, method_or_framework, "
        "specific_concepts, assumptions_limits, case_mapping, protocol_controls, "
        "misuse_or_overclaim_warnings, citation_snippets.\n\n"
        "Requirements:\n"
        "- paper_central_claim: 2-4 sentences about what this paper specifically argues.\n"
        "- specific_concepts: 5-10 concrete concepts, not generic course labels.\n"
        "- case_mapping: explain exactly how this paper changes the target project design.\n"
        "- protocol_controls: concrete controls or decision rules inspired by the paper.\n"
        "- misuse_or_overclaim_warnings: how not to misuse this paper in the target project.\n"
        "- citation_snippets: short paraphrased anchors with page hints if available; no long quotes.\n"
        "- Include compact citations using the short path, e.g. M04_.../Example.pdf.\n\n"
        f"Attempt: {attempt}. If the previous attempt failed, be stricter: output one complete JSON object and no prose.\n\n"
        f"Reading text:\n{text}"
    )
    token_budget = cfg.card_max_tokens * attempt
    raw = llm.complete(system, user, temperature=0.1, max_tokens=token_budget)
    obj = _parse_json_object(raw)
    if not raw.strip():
        obj["validation_error"] = "empty model output"
    elif not isinstance(obj, dict):
        obj = {"status": "parse_failed", "raw_model_output": raw, "validation_error": "not a JSON object"}
    obj.setdefault("short_path", paper["short_path"])
    obj.setdefault("module_id", paper["module_id"])
    obj.setdefault("module_name", paper["module_name"])
    obj["raw_model_output"] = obj.get("raw_model_output", raw)
    return obj


def _is_valid_card(card: dict) -> bool:
    if not isinstance(card, dict):
        return False
    raw = str(card.get("raw_model_output", "") or "")
    if card.get("status") == "parse_failed" and not raw.strip():
        card["validation_error"] = "empty parse-failed output"
        return False
    missing = []
    for field in REQUIRED_CARD_FIELDS:
        value = card.get(field)
        if value is None:
            missing.append(field)
        elif isinstance(value, str) and not value.strip():
            missing.append(field)
        elif isinstance(value, list) and not value:
            missing.append(field)
    if missing:
        card["validation_error"] = "missing or empty fields: " + ", ".join(missing)
        return False
    return True


def _parse_json_object(raw: str) -> dict:
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
    return {
        "status": "parse_failed",
        "raw_model_output": raw,
    }


def _load_cards(cards_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cards_dir.glob("*.json"))
    ]


def _synthesize_protocol_with_retries(llm, cards: list[dict], cfg: SemanticSweepConfig) -> str:
    for attempt in range(1, 4):
        synthesis = _synthesize_protocol(llm, cards, cfg, attempt=attempt)
        if synthesis.strip():
            return synthesis
        print(f"  ! synthesis empty on attempt {attempt}")
    raise RuntimeError("Semantic synthesis failed: empty model output after 3 attempts")


def _synthesize_protocol(llm, cards: list[dict], cfg: SemanticSweepConfig, *, attempt: int = 1) -> str:
    compact_cards = []
    for card in cards:
        compact_cards.append({
            "short_path": card.get("short_path"),
            "module": f"{card.get('module_id')} {card.get('module_name')}",
            "central_claim": card.get("paper_central_claim"),
            "concepts": card.get("specific_concepts"),
            "case_mapping": card.get("case_mapping") or card.get("dprk_case_mapping"),
            "protocol_controls": card.get("protocol_controls"),
            "warnings": card.get("misuse_or_overclaim_warnings"),
        })
    system = (
        "You are synthesizing an Assured Autonomy semantic network into a concrete "
        "project protocol. Use the paper cards substantively. Avoid "
        "generic syllabus summaries."
    )
    user = (
        f"Target project:\n{cfg.case_context}\n\n"
        "Semantic cards:\n"
        f"{json.dumps(compact_cards, indent=2)}\n\n"
        "Produce a coherent protocol, not a list of generic summaries. Required sections:\n"
        "1. Executive framing and research question.\n"
        "2. Semantic network: concept clusters and how papers connect to one another.\n"
        "3. System architecture and data flow.\n"
        "4. Runtime monitoring, fallback, and human textual oversight controls.\n"
        "5. Testing/evaluation plan and metrics.\n"
        "6. Legal, policy, privacy, and overclaim controls.\n"
        "7. Weak analogies and failure modes to avoid.\n"
        "8. Concise project-proposal summary.\n\n"
        "Cite compact paths inline, e.g. `M04_AnomalyAndFaultDetection/Chandola-2009-...pdf`."
    )
    return llm.complete(
        system,
        user,
        temperature=0.1,
        max_tokens=cfg.synthesis_max_tokens * attempt,
    )


def _render_card_md(card: dict) -> str:
    lines = [
        f"# {card.get('short_path', 'Semantic Card')}",
        "",
        f"Module: {card.get('module_id')} {card.get('module_name')}",
        "",
        "## Central Claim",
        str(card.get("paper_central_claim", "")),
        "",
        "## Method Or Framework",
        str(card.get("method_or_framework", "")),
        "",
        "## Specific Concepts",
        json.dumps(card.get("specific_concepts", []), indent=2),
        "",
        "## DPRK Case Mapping",
        str(card.get("case_mapping", card.get("dprk_case_mapping", ""))),
        "",
        "## Protocol Controls",
        json.dumps(card.get("protocol_controls", []), indent=2),
        "",
        "## Warnings",
        json.dumps(card.get("misuse_or_overclaim_warnings", []), indent=2),
        "",
    ]
    return "\n".join(lines)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "default"


def _safe_stem(value: str) -> str:
    return _safe_name(value.replace("\\", "__").replace("/", "__").replace(".pdf", ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", default='papers/papers')
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-doc-chars", type=int, default=45000)
    parser.add_argument("--card-max-tokens", type=int, default=2400)
    parser.add_argument("--synthesis-max-tokens", type=int, default=7000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--out-dir", default="semantic_runs")
    parser.add_argument("--modules", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--synthesize-only", action="store_true")
    parser.add_argument("--case-file", help="Read target project/case context from a UTF-8 text file")
    parser.add_argument("--case-text", help="Inline target project/case context")
    parser.add_argument("--case-name", default="dprk_protocol")
    parser.add_argument("--card-retries", type=int, default=2)
    parser.add_argument("--allow-failed-cards", action="store_true")
    args = parser.parse_args(argv)
    case_context = DPRK_CASE
    if args.case_file:
        case_context = Path(args.case_file).read_text(encoding="utf-8")
    elif args.case_text:
        case_context = args.case_text
    cfg = SemanticSweepConfig(
        papers_dir=Path(args.papers_dir),
        provider=args.provider,
        model=args.model,
        max_doc_chars=args.max_doc_chars,
        card_max_tokens=args.card_max_tokens,
        synthesis_max_tokens=args.synthesis_max_tokens,
        sleep_seconds=args.sleep_seconds,
        out_dir=Path(args.out_dir),
        modules=args.modules,
        limit=args.limit,
        force=args.force,
        synthesize_only=args.synthesize_only,
        case_context=case_context,
        case_name=args.case_name,
        card_retries=args.card_retries,
        allow_failed_cards=args.allow_failed_cards,
    )
    run_semantic_sweep(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

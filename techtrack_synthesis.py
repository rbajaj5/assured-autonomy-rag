"""Synthesize a TechTrack-informed Assured Autonomy experiment plan."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import get_llm


DEFAULT_TECHTRACK_DIR = Path("external_repos/techtrack")
DEFAULT_CARD_DIR = Path(
    'semantic_runs'
    r"\semantic_sweep_gpt-5-5_20260508-171535-277395\cards"
)


@dataclass
class TechTrackSynthesisConfig:
    techtrack_dir: Path = DEFAULT_TECHTRACK_DIR
    card_dir: Path = DEFAULT_CARD_DIR
    provider: str = "openai"
    model: str = "gpt-5.5"
    max_card_chars: int = 240
    max_techtrack_chars: int = 45000
    max_tokens: int = 9000
    out_dir: Path = Path("techtrack_synthesis_runs")
    modules: list[str] | None = None
    max_cards: int | None = 24


def run_techtrack_synthesis(cfg: TechTrackSynthesisConfig) -> dict:
    run_dir = _new_run_dir(cfg.out_dir, cfg.model)
    run_dir.mkdir(parents=True, exist_ok=False)

    techtrack = _load_techtrack_context(cfg.techtrack_dir, cfg.max_techtrack_chars)
    cards = _load_compact_cards(
        cfg.card_dir,
        cfg.max_card_chars,
        modules=cfg.modules,
        max_cards=cfg.max_cards,
    )
    spectral = _read_optional(cfg.techtrack_dir / "reports" / "assets" / "spectral_audit" / "spectral_audit.md")

    prompt_payload = {
        "techtrack_context": techtrack,
        "spectral_audit": spectral,
        "assured_autonomy_cards": cards,
    }
    (run_dir / "prompt_payload.json").write_text(
        json.dumps(prompt_payload, indent=2),
        encoding="utf-8",
    )

    system = (
        "You are an expert AI research engineer designing an Assured Autonomy course project. "
        "You must use TechTrack as the concrete experimental scaffold and the course-paper cards "
        "as the assurance theory. Be specific, testable, and skeptical. Do not make the project "
        "about legal RAG or LLM runtime control. The scientific core is training competing "
        "YOLO/CNN perception models, probing capacity allocation with spectral diagnostics, "
        "and designing cost-aware assurance tests."
    )
    user = _build_prompt(prompt_payload)

    llm = get_llm(cfg.provider, model=cfg.model)
    answer = _complete_with_retries(llm, system, user, cfg.max_tokens)

    result = {
        "run_dir": str(run_dir),
        "techtrack_dir": str(cfg.techtrack_dir),
        "card_dir": str(cfg.card_dir),
        "provider": cfg.provider,
        "model": cfg.model,
        "answer": answer,
    }
    (run_dir / "techtrack_assurance_synthesis.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (run_dir / "techtrack_assurance_synthesis.md").write_text(answer, encoding="utf-8")
    try:
        print(answer)
    except UnicodeEncodeError:
        import sys
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(answer.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    print(f"\nSaved TechTrack synthesis directory: {run_dir}")
    return result


def _load_techtrack_context(root: Path, max_chars: int) -> dict:
    files = [
        "ANALYSIS_DESIGN_ASSIGNMENT.md",
        "IMPLEMENTATION_ASSIGNMENT.md",
        "CASE_ANALYSIS.md",
        "techtrack/modules/inference/model.py",
        "techtrack/modules/inference/nms.py",
        "techtrack/modules/rectification/hard_negative_mining.py",
        "techtrack/modules/rectification/augmentation.py",
        "tools/export_preds_darknet.py",
        "tools/coco_eval.py",
        "tools/spectral_audit_darknet.py",
    ]
    data = {}
    remaining = max_chars
    for rel in files:
        if remaining <= 0:
            break
        path = root / rel
        text = _read_optional(path)
        if not text:
            continue
        text = _trim(text, min(remaining, 9000))
        data[rel] = text
        remaining -= len(text)
    return data


def _load_compact_cards(card_dir: Path, max_card_chars: int, *,
                        modules: list[str] | None = None,
                        max_cards: int | None = None) -> list[dict]:
    wanted = set(modules or [])
    cards = []
    for path in sorted(card_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        module_id = obj.get("module_id")
        if wanted and module_id not in wanted:
            continue
        cards.append({
            "short_path": obj.get("short_path"),
            "module": f"{obj.get('module_id')} {obj.get('module_name')}",
            "central_claim": _trim(_stringify(obj.get("paper_central_claim")), max_card_chars),
            "concepts": obj.get("specific_concepts", [])[:8],
            "case_mapping": _trim(_stringify(obj.get("case_mapping")), max_card_chars),
            "warnings": _trim(_stringify(obj.get("misuse_or_overclaim_warnings")), max_card_chars),
        })
        if max_cards and len(cards) >= max_cards:
            break
    return cards


def _build_prompt(payload: dict) -> str:
    return (
        "Task:\n"
        "Design the next experiment plan by translating TechTrack into an Assured Autonomy "
        "project about trained YOLO/CNN competitors for German traffic-sign perception. "
        "Use the course cards to decide what to test, what not to overclaim, and how to "
        "control test cost.\n\n"
        "Current project framing:\n"
        "- TechTrack originally evaluated given YOLOv4-tiny models; our project will train competitors.\n"
        "- The competitors should vary capacity allocation, geometric priors, training objectives, or robustness recipes.\n"
        "- The target domain is German traffic-sign perception, preferably GTSRB/GTSDB-style data.\n"
        "- Spectral diagnostics are a weak/cheap triage signal, not assurance by themselves.\n"
        "- Hard negative mining and TechTrack metrics are empirical feedback signals.\n"
        "- We want cost-aware testing: use cheap audits to decide where expensive robustness/adversarial/verification tests are worth running.\n\n"
        "Required output sections:\n"
        "1. Updated project title and one-paragraph thesis.\n"
        "2. Mapping from TechTrack tasks to German traffic-sign assurance tasks.\n"
        "3. Candidate model competitors to train, with the capacity/geometric variable each one probes.\n"
        "4. Spectral diagnostics to compute and what capacity allocation each metric may indicate.\n"
        "5. TechTrack-style metrics to keep, metrics to change, and new metrics needed.\n"
        "6. Hard-negative mining role and how it interacts with spectra.\n"
        "7. Adversarial/shift tests to add, prioritized by cost and course justification.\n"
        "8. Cost-aware assurance test policy: cheap screen -> targeted test -> expensive verification.\n"
        "9. Corpus-grounded rubric for grading model readiness, with citations to compact paper paths.\n"
        "10. Minimum viable experiment and stretch experiment.\n"
        "11. Paper-presentation recommendation: required reading plus outside newer paper.\n"
        "12. Risks, weak analogies, and overclaims to avoid.\n\n"
        "Important constraints:\n"
        "- Do not center the project on LLMs, legal RAG, or passenger input.\n"
        "- Do not claim spectral diagnostics prove safety.\n"
        "- Cite course papers using compact paths from the cards.\n"
        "- Be concrete enough that we can turn this into an experiment proposal.\n\n"
        "Inputs:\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _complete_with_retries(llm, system: str, user: str, max_tokens: int) -> str:
    for attempt in range(1, 4):
        out = llm.complete(system, user, temperature=0.1, max_tokens=max_tokens * attempt)
        if out.strip():
            return out
        print(f"  ! TechTrack synthesis empty on attempt {attempt}")
    raise RuntimeError("TechTrack synthesis failed: empty model output after 3 attempts")


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _trim(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _new_run_dir(base: Path, model: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in model)
    return base / f"techtrack_synthesis_{safe_model}_{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--techtrack-dir", default=str(DEFAULT_TECHTRACK_DIR))
    parser.add_argument("--card-dir", default=str(DEFAULT_CARD_DIR))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-card-chars", type=int, default=240)
    parser.add_argument("--max-techtrack-chars", type=int, default=45000)
    parser.add_argument("--max-tokens", type=int, default=9000)
    parser.add_argument("--out-dir", default="techtrack_synthesis_runs")
    parser.add_argument("--modules", nargs="*",
                        default=["M04", "M05", "M06", "M07", "M11", "M12"])
    parser.add_argument("--max-cards", type=int, default=24)
    args = parser.parse_args(argv)
    cfg = TechTrackSynthesisConfig(
        techtrack_dir=Path(args.techtrack_dir),
        card_dir=Path(args.card_dir),
        provider=args.provider,
        model=args.model,
        max_card_chars=args.max_card_chars,
        max_techtrack_chars=args.max_techtrack_chars,
        max_tokens=args.max_tokens,
        out_dir=Path(args.out_dir),
        modules=args.modules,
        max_cards=args.max_cards,
    )
    run_techtrack_synthesis(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

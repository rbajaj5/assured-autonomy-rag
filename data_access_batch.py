"""Run Assured Autonomy concept application over restricted dataset access cases."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from .concept_debate import apply_concepts_to_policy


DEFAULT_CASES = Path("data/restricted_dataset_access_cases.json")
DEFAULT_TEMPLATE = Path("prompts/data_access_assurance_template.txt")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "case"


def _load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a JSON list")
    for case in cases:
        if not case.get("id") or not case.get("name") or not case.get("urls"):
            raise ValueError(f"Case is missing id, name, or urls: {case}")
    return cases


def _render_question(template: str, case: dict) -> str:
    return template.format(
        dataset_name=case["name"],
        dataset_focus=case.get("focus", "restricted academic dataset"),
    )


def run_batch(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    template_path = Path(args.template)
    concept_run = Path(args.concept_run)
    cases = _load_cases(cases_path)
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case["id"] in wanted]
    template = template_path.read_text(encoding="utf-8")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    root = Path(args.out_dir) / f"dataset_access_batch_{_safe_name(args.provider)}_{_safe_name(args.model)}_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    manifest: dict = {
        "created_at": stamp,
        "concept_run": str(concept_run),
        "provider": args.provider,
        "model": args.model,
        "cases_path": str(cases_path),
        "template_path": str(template_path),
        "results": [],
    }

    for i, case in enumerate(cases, start=1):
        case_dir = root / f"{i:02d}_{_safe_name(case['id'])}"
        case_dir.mkdir(parents=True, exist_ok=False)
        question = _render_question(template, case)
        (case_dir / "question.txt").write_text(question, encoding="utf-8")
        (case_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
        print(f"\n[{i}/{len(cases)}] {case['name']}")

        row = {
            "id": case["id"],
            "name": case["name"],
            "urls": case["urls"],
            "status": "started",
            "case_dir": str(case_dir),
            "attempts": [],
        }
        manifest["results"].append(row)
        _write_manifest(root, manifest)

        for attempt in range(args.retries + 1):
            budget = max(args.min_tokens, int(args.max_tokens * (0.75 ** attempt)))
            try:
                result = apply_concepts_to_policy(
                    concept_run=concept_run,
                    policy_urls=case["urls"],
                    provider=args.provider,
                    model=args.model,
                    question=question,
                    max_policy_chars=args.max_policy_chars,
                    max_tokens=budget,
                    out_dir=case_dir,
                )
                row["status"] = "complete"
                row["attempts"].append({"attempt": attempt + 1, "max_tokens": budget, "status": "complete"})
                row["answer_chars"] = len(result.get("answer", ""))
                break
            except Exception as exc:
                row["attempts"].append({
                    "attempt": attempt + 1,
                    "max_tokens": budget,
                    "status": "failed",
                    "error": repr(exc),
                })
                row["status"] = "failed"
                row["error"] = repr(exc)
                print(f"  ! failed attempt {attempt + 1} with {budget} tokens: {exc}")
                if attempt < args.retries:
                    time.sleep(args.retry_sleep)
            finally:
                _write_manifest(root, manifest)

    _write_summary(root, manifest)
    print(f"\nSaved dataset access batch directory: {root}")
    return 0


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_summary(root: Path, manifest: dict) -> None:
    lines = [
        "# Restricted Dataset Access Batch",
        "",
        f"Concept run: `{manifest['concept_run']}`",
        f"Provider/model: `{manifest['provider']}` / `{manifest['model']}`",
        "",
        "| Case | Status | Attempts | Saved under |",
        "|---|---:|---:|---|",
    ]
    for row in manifest["results"]:
        lines.append(
            f"| {row['name']} | {row['status']} | {len(row['attempts'])} | `{row['case_dir']}` |"
        )
    lines.append("")
    lines.append("Each case directory contains the rendered question, source case metadata, and one `concept_apply_*` run directory if the model completed.")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-run", required=True, help="Path to concept_debate_*.json")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--only", nargs="*", help="Optional case ids to run")
    parser.add_argument("--provider", default="mistral", choices=["openai", "mistral", "anthropic", "ollama", "stub", "peft-local"])
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--max-policy-chars", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--min-tokens", type=int, default=1400)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--out-dir", default="concept_runs")
    return run_batch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

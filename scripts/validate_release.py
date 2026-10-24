"""Run public, offline checks; distinguish excluded corpus integrations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("ASSURED_RAG_PRIVATE_CORPUS_TESTS", None)
    for key in list(env):
        if key.endswith("API_KEY") or key in {"OPENAI_ACCESS_TOKEN", "HF_TOKEN"}:
            env.pop(key)
    cases = {
        "assurance_regressions": ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"],
        "synthetic_pipeline": ["test_pipeline.py"],
        "aate_cli": ["-m", "aate", "--help"],
        "offline_example": ["-m", "assured_rag.evaluate", "tests/fixtures/proposals/safe_complete_offline_log_replay.txt",
                            "--corpus-dir", "docs/safe_assurance_evaluator.md", "--no-embeddings", "--format", "json"],
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    results = {}
    for name, args in cases.items():
        done = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=240)
        output = done.stdout + "\n" + done.stderr
        (ROOT / "reports" / f"validation_{name}.log").write_text(output, encoding="utf-8")
        result = {"passed": done.returncode == 0, "exit_code": done.returncode}
        if name == "assurance_regressions":
            count = re.search(r"Ran (\d+) tests", output)
            skipped = re.search(r"skipped=(\d+)", output)
            result.update(tests_discovered=int(count[1]) if count else None, skipped_private_corpus=int(skipped[1]) if skipped else 0)
        if name == "offline_example" and done.returncode == 0:
            payload = json.loads(done.stdout)
            result.update(verdict=payload["verdict"], citation_count=len(payload["supporting_citations"]))
        results[name] = result
        print(name, result)
        if done.returncode:
            print(output[-3000:])
    summary = {"passed": all(r["passed"] for r in results.values()), "validation_scope": "Offline source-release checks, not model-quality or corpus replication",
               "paid_api_calls": False, "model_weights_downloaded": False, "checks": results,
               "not_tested": ["four private-corpus integrations", "live provider availability", "semantic answer quality", "optional PEFT training", "Linux/macOS execution"]}
    (ROOT / "reports/public_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

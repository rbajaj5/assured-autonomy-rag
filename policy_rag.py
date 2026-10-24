"""Policy-aware wrapper around iterative RAG."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .iterative_rag import IterativeRagConfig, IterativeRagRunner
from .loaders import NewItem, load_new_item


DEFAULT_POLICY_QUESTION = (
    "Dissect this policy for academic access and possible dataset use. Identify "
    "eligibility requirements, academic-only restrictions, dataset-use permissions "
    "and prohibitions, redistribution limits, automated extraction or model-training "
    "limits, privacy/security obligations, accountability obligations, ambiguity/gap "
    "areas, likely denial reasons, and evidence a JHU student could provide to reduce "
    "each concern. Do not infer permissions that are not present in the policy text."
)


@dataclass
class PolicyRagConfig:
    iterative: IterativeRagConfig
    max_policy_chars: int = 30000


def load_policy_source(*, policy_url: str | None = None,
                       policy_file: str | None = None,
                       policy_text: str | None = None) -> NewItem:
    provided = [x for x in (policy_url, policy_file, policy_text) if x]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of --policy-url, --policy-file, or --policy-text.")
    return load_new_item(policy_url or policy_file or policy_text or "")


def load_policy_sources(*, policy_urls: list[str] | None = None,
                        policy_file: str | None = None,
                        policy_text: str | None = None) -> list[NewItem]:
    provided = int(bool(policy_urls)) + int(bool(policy_file)) + int(bool(policy_text))
    if provided != 1:
        raise ValueError("Provide --policy-url one or more times, or exactly one of --policy-file/--policy-text.")
    if policy_urls:
        return [load_new_item(url) for url in policy_urls]
    return [load_new_item(policy_file or policy_text or "")]


def build_policy_prompt(item: NewItem, question: str,
                        max_policy_chars: int = 30000) -> str:
    return build_multi_policy_prompt([item], question, max_policy_chars=max_policy_chars)


def build_multi_policy_prompt(items: list[NewItem], question: str,
                              max_policy_chars: int = 30000) -> str:
    remaining = max_policy_chars
    source_blocks: list[str] = []
    for i, item in enumerate(items, start=1):
        if remaining <= 0:
            policy = ""
        else:
            policy = item.text[:remaining]
        remaining -= len(policy)
        truncated = len(item.text) > len(policy)
        trunc_note = (
            f"\n\n[Source {i} text truncated to {len(policy)} of {len(item.text)} characters.]"
            if truncated else ""
        )
        source_blocks.append(
            f"### Policy/source S{i}\n"
            f"{item.header()}\n\n"
            f"Text:\n---\n{policy}\n---{trunc_note}"
        )
    sources_text = "\n\n".join(source_blocks)
    return (
        f"Policy/source materials:\n{sources_text}\n\n"
        f"User question:\n{question}\n\n"
        "Task instructions:\n"
        "- First parse each supplied source into concrete clauses, obligations, prohibitions, and ambiguities.\n"
        "- Compare sources when they differ; identify which source supports each key factual claim.\n"
        "- Cite supplied policy/source pages as [S1], [S2], etc. Do not cite them with [CITE_N].\n"
        "- Use [CITE_N] only for Assured Autonomy corpus evidence provided later by the retriever.\n"
        "- Then map each clause to RAG-addressable Assured Autonomy concepts.\n"
        "- Use the Assured Autonomy corpus for governance/safety/security/privacy/verification reasoning.\n"
        "- Distinguish what the supplied sources explicitly say from what the course corpus suggests as risk analysis.\n"
        "- Do not claim Jureko-specific facts beyond the supplied policy text.\n"
    )


def run_policy_rag(*, cfg: PolicyRagConfig,
                   policy_url: str | None = None,
                   policy_urls: list[str] | None = None,
                   policy_file: str | None = None,
                   policy_text: str | None = None,
                   question: str = DEFAULT_POLICY_QUESTION) -> dict:
    urls = policy_urls or ([policy_url] if policy_url else None)
    items = load_policy_sources(
        policy_urls=urls,
        policy_file=policy_file,
        policy_text=policy_text,
    )
    prompt = build_multi_policy_prompt(items, question, max_policy_chars=cfg.max_policy_chars)
    result = IterativeRagRunner(cfg.iterative).run(prompt)
    result["answer"] = _expand_policy_source_refs(result.get("answer", ""), items)
    result["critique"] = result.get("critique", {})
    result["policy_source"] = [
        {
            "source_kind": item.source_kind,
            "source": item.source,
            "title": item.title,
            "text_len": len(item.text),
            "max_policy_chars_total": cfg.max_policy_chars,
        }
        for item in items
    ]
    _save_policy_result(result, cfg.iterative.out_dir)
    return result


def _expand_policy_source_refs(answer: str, items: list[NewItem]) -> str:
    source_refs = {
        f"S{i}": f"{item.title or 'Policy source'} ({item.source})"
        for i, item in enumerate(items, start=1)
    }

    def replace_s(match: re.Match[str]) -> str:
        key = match.group(1)
        return f"[{source_refs.get(key, key)}]"

    answer = re.sub(r"\[(S\d+)\]", replace_s, answer)

    # If the model incorrectly used high CITE numbers for policy pages, map the
    # final N supplied sources in the prompt back to their policy/source labels.
    offset = max(0, 18 - len(items))

    def replace_stray_cite(match: re.Match[str]) -> str:
        n = int(match.group(1))
        source_idx = n - offset
        key = f"S{source_idx}"
        if key in source_refs:
            return f"[{source_refs[key]}]"
        return match.group(0)

    return re.sub(r"\[CITE_(\d+)\]", replace_stray_cite, answer)


def _save_policy_result(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = out_dir / f"policy_rag_{_safe_name(result['provider'])}_{_safe_name(result['model'] or 'default')}_{stamp}"
    json_path = Path(f"{stem}.json")
    md_path = Path(f"{stem}.md")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        "# Policy RAG Run",
        "",
        "## Policy Source",
        "```json",
        json.dumps(result["policy_source"], indent=2),
        "```",
        "",
        "## Answer",
        result["answer"],
        "",
        "## Critique",
        "```json",
        json.dumps(result["critique"], indent=2),
        "```",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    return safe.strip("-") or "default"

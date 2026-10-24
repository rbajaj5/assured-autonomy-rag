"""SIRA-style discriminative retrieval sweep for Assured Autonomy readings.

This is inspired by the SIRA paper's core retrieval idea: spend a small amount
of cognition up front to create corpus-discriminative vocabulary, filter it with
document-frequency statistics, and retrieve high-margin evidence before doing
expensive synthesis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import AA_MODULES, module_id_from_dirname
from .llm import get_llm
from .pdf_text import extract_pdf_pages


DEFAULT_CASE = (
    "Upgrade TechTrack into an Assured Autonomy experiment for German traffic-sign "
    "perception. Train competing YOLO/CNN models and study how capacity allocation, "
    "geometric priors, robust objectives, hard-negative mining, spectral diagnostics, "
    "dataset shift tests, adversarial tests, calibration, latency, and bounded "
    "verification affect readiness. Use spectra as triage, not proof."
)

VOCAB_FIELDS = (
    "technical_terms",
    "methods",
    "metrics",
    "failure_modes",
    "techtrack_bridge_terms",
    "confusers_to_avoid",
)

PAGE_CARD_FIELDS = (
    "evidence_claims",
    "methods_or_tests",
    "concepts",
    "techtrack_relevance",
    "capacity_or_geometry_hooks",
    "assurance_metrics",
    "limits_and_overclaims",
    "page_anchors",
)

PAPER_CARD_FIELDS = (
    "paper_actual_contribution",
    "evidence_used",
    "techtrack_upgrade_role",
    "capacity_allocation_role",
    "assurance_test_role",
    "what_not_to_claim",
    "study_notes",
    "citation_anchors",
)


@dataclass
class SiraSweepConfig:
    papers_dirs: list[Path]
    provider: str = "openai"
    model: str = "gpt-5.5"
    out_dir: Path = Path("sira_sweep_runs")
    resume_dir: Path | None = None
    case_file: Path | None = None
    case_text: str | None = None
    modules: list[str] | None = None
    limit: int | None = None
    max_pages_per_pdf: int = 500
    sample_chars: int = 18000
    top_pages_per_paper: int = 8
    max_selected_page_chars: int = 18000
    max_df_ratio: float = 0.45
    min_df: int = 1
    query_weight: float = 3.0
    document_weight: float = 0.85
    vocab_max_tokens: int = 1600
    page_card_max_tokens: int = 2600
    paper_card_max_tokens: int = 4200
    synthesis_max_tokens: int = 9000
    retries: int = 2
    sleep_seconds: float = 0.0
    force: bool = False
    retrieval_only: bool = False
    synthesize_only: bool = False


def run_sira_sweep(cfg: SiraSweepConfig) -> dict:
    run_dir = cfg.resume_dir or _new_run_dir(cfg.out_dir, cfg.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir = run_dir / "retrieval_plans"
    page_dir = run_dir / "page_cards"
    paper_dir = run_dir / "paper_cards"
    retrieval_dir.mkdir(exist_ok=True)
    page_dir.mkdir(exist_ok=True)
    paper_dir.mkdir(exist_ok=True)

    papers = _discover_papers(cfg.papers_dirs, cfg.modules)
    if cfg.limit:
        papers = papers[: cfg.limit]
    case_context = _case_context(cfg)
    (run_dir / "case_context.txt").write_text(case_context, encoding="utf-8")

    manifest = {
        "run_dir": str(run_dir),
        "provider": cfg.provider,
        "model": cfg.model,
        "papers_dirs": [str(p) for p in cfg.papers_dirs],
        "papers_total": len(papers),
        "papers": [],
        "method": "SIRA-style vocabulary enrichment + DF filtering + weighted BM25 page selection",
    }

    page_docs = _load_page_docs(papers, cfg.max_pages_per_pdf)
    df_stats = _document_frequencies(page_docs)
    (run_dir / "corpus_df.json").write_text(json.dumps(_compact_df(df_stats), indent=2), encoding="utf-8")

    llm = get_llm(cfg.provider, model=cfg.model)
    query_vocab_path = run_dir / "query_vocabulary.json"
    if query_vocab_path.exists() and not cfg.force:
        query_vocab = json.loads(query_vocab_path.read_text(encoding="utf-8"))
    else:
        query_vocab = _build_query_vocabulary(llm, case_context, cfg)
        query_vocab_path.write_text(json.dumps(query_vocab, indent=2), encoding="utf-8")
    query_terms = _validated_terms(_flatten_vocab(query_vocab), df_stats, cfg)
    (run_dir / "validated_query_terms.json").write_text(json.dumps(query_terms, indent=2), encoding="utf-8")

    if not cfg.synthesize_only:
        for i, paper in enumerate(papers, start=1):
            print(f"[{i}/{len(papers)}] sira-select {paper['short_path']}")
            safe = _safe_stem(paper["short_path"])
            retrieval_path = retrieval_dir / f"{safe}.json"
            if retrieval_path.exists() and not cfg.force:
                plan = json.loads(retrieval_path.read_text(encoding="utf-8"))
            else:
                paper_pages = [doc for doc in page_docs if doc["paper_index"] == paper["paper_index"]]
                doc_vocab = _build_document_vocabulary_with_retries(llm, paper, paper_pages, case_context, cfg)
                doc_terms = _validated_terms(_flatten_vocab(doc_vocab), df_stats, cfg)
                ranked = _rank_pages(paper_pages, query_terms, doc_terms, df_stats, cfg)
                plan = {
                    "short_path": paper["short_path"],
                    "module_id": paper["module_id"],
                    "module_name": paper["module_name"],
                    "document_vocabulary": doc_vocab,
                    "validated_document_terms": doc_terms,
                    "validated_query_terms": query_terms,
                    "selected_pages": ranked[: cfg.top_pages_per_paper],
                }
                retrieval_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                if cfg.sleep_seconds:
                    time.sleep(cfg.sleep_seconds)

            if cfg.retrieval_only:
                manifest["papers"].append({
                    "short_path": paper["short_path"],
                    "selected_pages": [p["page"] for p in plan.get("selected_pages", [])],
                    "retrieval_plan": str(retrieval_path),
                })
                continue

            page_card_path = page_dir / f"{safe}.json"
            if page_card_path.exists() and not cfg.force:
                page_card = json.loads(page_card_path.read_text(encoding="utf-8"))
            else:
                page_card = _build_page_card_with_retries(llm, paper, plan, case_context, cfg)
                page_card_path.write_text(json.dumps(page_card, indent=2), encoding="utf-8")
                if cfg.sleep_seconds:
                    time.sleep(cfg.sleep_seconds)

            paper_card_path = paper_dir / f"{safe}.json"
            if paper_card_path.exists() and not cfg.force:
                paper_card = json.loads(paper_card_path.read_text(encoding="utf-8"))
            else:
                paper_card = _build_paper_card_with_retries(llm, paper, plan, page_card, case_context, cfg)
                paper_card_path.write_text(json.dumps(paper_card, indent=2), encoding="utf-8")
                (paper_dir / f"{safe}.md").write_text(_render_paper_card_md(paper_card), encoding="utf-8")

            manifest["papers"].append({
                "short_path": paper["short_path"],
                "selected_pages": [p["page"] for p in plan.get("selected_pages", [])],
                "retrieval_plan": str(retrieval_path),
                "page_card": str(page_card_path),
                "paper_card": str(paper_card_path),
            })
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paper_cards = _load_cards(paper_dir)
    synthesis = ""
    if paper_cards:
        synthesis = _synthesize_network(llm, paper_cards, case_context, cfg)
        (run_dir / "sira_synthesis.md").write_text(synthesis, encoding="utf-8")
        (run_dir / "sira_synthesis.json").write_text(
            json.dumps({"paper_card_count": len(paper_cards), "synthesis": synthesis}, indent=2),
            encoding="utf-8",
        )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved SIRA sweep directory: {run_dir}")
    return {"manifest": manifest, "paper_card_count": len(paper_cards), "synthesis": synthesis}


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
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = Path(path.name)
            rows.append({
                "paper_index": len(rows),
                "path": path,
                "short_path": str(rel).replace("\\", "/"),
                "module_id": module_id,
                "module_name": module_name,
                "filename": path.name,
            })
    return rows


def _load_page_docs(papers: list[dict], max_pages: int) -> list[dict]:
    docs = []
    for paper in papers:
        pages = extract_pdf_pages(paper["path"], max_pages=max_pages)
        for i, text in enumerate(pages, start=1):
            cleaned = _clean(text)
            if not cleaned:
                continue
            docs.append({
                "paper_index": paper["paper_index"],
                "short_path": paper["short_path"],
                "module_id": paper["module_id"],
                "module_name": paper["module_name"],
                "page": i,
                "text": cleaned,
                "tokens": _tokens(cleaned),
            })
    return docs


def _document_frequencies(page_docs: list[dict]) -> dict:
    df = Counter()
    total = 0
    for doc in page_docs:
        total += 1
        df.update(set(doc["tokens"]))
    return {"total_docs": total, "df": dict(df)}


def _compact_df(stats: dict, limit: int = 5000) -> dict:
    items = sorted(stats["df"].items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return {"total_docs": stats["total_docs"], "top_terms": dict(items)}


def _build_query_vocabulary(llm, case_context: str, cfg: SiraSweepConfig) -> dict:
    system = (
        "You create corpus-discriminative retrieval vocabulary. Return valid JSON only. "
        "Prefer technical evidence terms that separate useful pages from generic AI/safety text."
    )
    user = (
        f"Project/query context:\n{case_context}\n\n"
        "Return JSON with these keys: "
        "technical_terms, methods, metrics, failure_modes, techtrack_bridge_terms, confusers_to_avoid. "
        "Each value must be a list of short terms or phrases. Include likely missing vocabulary from "
        "Assured Autonomy papers: e.g. terms an expert would search for even if not in the user query. "
        "Avoid generic terms unless paired with a discriminative modifier."
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=cfg.vocab_max_tokens)
    obj = _parse_json(raw, defaults={field: [] for field in VOCAB_FIELDS})
    if not any(obj.get(field) for field in VOCAB_FIELDS):
        obj["technical_terms"] = SEED_QUERY_TERMS
    return obj


def _build_document_vocabulary_with_retries(llm, paper: dict, pages: list[dict], case_context: str, cfg: SiraSweepConfig) -> dict:
    sample = _paper_sample(pages, cfg.sample_chars)
    last = {}
    for attempt in range(1, cfg.retries + 2):
        raw = _build_document_vocabulary(llm, paper, sample, case_context, cfg, attempt)
        obj = _parse_json(raw, defaults={field: [] for field in VOCAB_FIELDS})
        if any(obj.get(field) for field in VOCAB_FIELDS):
            return obj
        print(f"  ! document vocab empty attempt {attempt}")
        last = obj
    last["technical_terms"] = _tokens(f"{paper['short_path']} {paper['module_name']}")[:20]
    return last


def _build_document_vocabulary(llm, paper: dict, sample: str, case_context: str, cfg: SiraSweepConfig, attempt: int) -> str:
    system = (
        "You enrich one PDF with missing search vocabulary for retrieval. Return valid JSON only. "
        "Do not summarize the paper. Produce terms that would help retrieve the paper's relevant evidence pages."
    )
    user = (
        f"Paper: {paper['short_path']}\n"
        f"Module: {paper['module_id']} {paper['module_name']}\n\n"
        f"Project/query context:\n{case_context}\n\n"
        "Return JSON with these keys: technical_terms, methods, metrics, failure_modes, "
        "techtrack_bridge_terms, confusers_to_avoid. Each value must be a list of short terms or phrases. "
        "Terms should be specific enough to separate this paper's useful evidence from corpus-level confusers. "
        f"Attempt: {attempt}\n\n"
        f"PDF sample:\n{sample}"
    )
    return llm.complete(system, user, temperature=0.1, max_tokens=cfg.vocab_max_tokens * attempt)


def _rank_pages(pages: list[dict], query_terms: list[dict], doc_terms: list[dict], df_stats: dict, cfg: SiraSweepConfig) -> list[dict]:
    docs = [{"tokens": p["tokens"]} for p in pages]
    avgdl = sum(len(d["tokens"]) for d in docs) / max(len(docs), 1)
    ranked = []
    weighted_terms = []
    for term in query_terms:
        weighted_terms.append((term["term"], cfg.query_weight, term))
    for term in doc_terms:
        weighted_terms.append((term["term"], cfg.document_weight, term))
    for page in pages:
        score = _bm25_score(page["tokens"], weighted_terms, df_stats, avgdl)
        matched = _matched_terms(page["text"], weighted_terms)
        ranked.append({
            "short_path": page["short_path"],
            "module_id": page["module_id"],
            "module_name": page["module_name"],
            "page": page["page"],
            "score": score,
            "matched_terms": matched[:30],
            "snippet": _snippet(page["text"], matched, limit=1200),
            "text": page["text"][: cfg.max_selected_page_chars],
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def _bm25_score(tokens: list[str], weighted_terms: list[tuple[str, float, dict]], df_stats: dict, avgdl: float) -> float:
    counts = Counter(tokens)
    dl = len(tokens)
    total_docs = max(int(df_stats["total_docs"]), 1)
    k1 = 1.4
    b = 0.72
    score = 0.0
    for term, weight, meta in weighted_terms:
        pieces = _tokens(term)
        if not pieces:
            continue
        tf = min(counts.get(piece, 0) for piece in pieces)
        if tf <= 0:
            continue
        df = max(int(meta.get("df", 1)), 1)
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
        score += weight * idf * (tf * (k1 + 1.0) / denom)
    return float(score)


def _validated_terms(raw_terms: list[str], df_stats: dict, cfg: SiraSweepConfig) -> list[dict]:
    seen = set()
    rows = []
    total_docs = max(int(df_stats["total_docs"]), 1)
    for raw in raw_terms:
        term = _normalize_term(raw)
        if not term or term in seen:
            continue
        seen.add(term)
        pieces = _tokens(term)
        if not pieces:
            continue
        dfs = [int(df_stats["df"].get(piece, 0)) for piece in pieces]
        df = min(dfs) if dfs else 0
        if df < cfg.min_df:
            continue
        ratio = df / total_docs
        if ratio > cfg.max_df_ratio:
            continue
        rows.append({"term": term, "df": df, "df_ratio": ratio})
    rows.sort(key=lambda r: (r["df_ratio"], r["term"]))
    return rows


def _flatten_vocab(vocab: dict) -> list[str]:
    terms = []
    for field in VOCAB_FIELDS:
        value = vocab.get(field, [])
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list):
            terms.extend(str(v) for v in value)
    return terms


def _build_page_card_with_retries(llm, paper: dict, plan: dict, case_context: str, cfg: SiraSweepConfig) -> dict:
    last = {}
    for attempt in range(1, cfg.retries + 2):
        card = _build_page_card(llm, paper, plan, case_context, cfg, attempt)
        if _valid(card, PAGE_CARD_FIELDS):
            card["status"] = "complete"
            return card
        print(f"  ! page card invalid attempt {attempt}: {card.get('validation_error', 'invalid')}")
        last = card
    last["status"] = "failed"
    return last


def _build_page_card(llm, paper: dict, plan: dict, case_context: str, cfg: SiraSweepConfig, attempt: int) -> dict:
    selected = [
        {
            "page": p.get("page"),
            "score": p.get("score"),
            "matched_terms": p.get("matched_terms"),
            "text": p.get("text", "")[: cfg.max_selected_page_chars],
        }
        for p in plan.get("selected_pages", [])
    ]
    system = (
        "You analyze selected high-margin pages from one academic PDF. Return valid JSON only. "
        "Anchor every substantive claim to page numbers. Avoid generic course-syllabus summaries."
    )
    user = (
        f"Paper: {paper['short_path']}\n"
        f"Module: {paper['module_id']} {paper['module_name']}\n\n"
        f"Project context:\n{case_context}\n\n"
        "These pages were selected by SIRA-style discriminative retrieval. Explain what they actually support. "
        "Return JSON with required keys: evidence_claims, methods_or_tests, concepts, techtrack_relevance, "
        "capacity_or_geometry_hooks, assurance_metrics, limits_and_overclaims, page_anchors. "
        "If evidence is weak for TechTrack, say so. Do not invent missing details.\n\n"
        f"Attempt: {attempt}\n\n"
        f"Selected pages:\n{json.dumps(selected, indent=2)}"
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=cfg.page_card_max_tokens * attempt)
    obj = _parse_json(raw, defaults={})
    obj.setdefault("short_path", paper["short_path"])
    obj.setdefault("module_id", paper["module_id"])
    obj.setdefault("module_name", paper["module_name"])
    obj.setdefault("selected_pages", [p.get("page") for p in plan.get("selected_pages", [])])
    obj["raw_model_output"] = obj.get("raw_model_output", raw)
    return obj


def _build_paper_card_with_retries(llm, paper: dict, plan: dict, page_card: dict, case_context: str, cfg: SiraSweepConfig) -> dict:
    last = {}
    for attempt in range(1, cfg.retries + 2):
        card = _build_paper_card(llm, paper, plan, page_card, case_context, cfg, attempt)
        if _valid(card, PAPER_CARD_FIELDS):
            card["status"] = "complete"
            return card
        print(f"  ! paper card invalid attempt {attempt}: {card.get('validation_error', 'invalid')}")
        last = card
    last["status"] = "failed"
    return last


def _build_paper_card(llm, paper: dict, plan: dict, page_card: dict, case_context: str, cfg: SiraSweepConfig, attempt: int) -> dict:
    compact_plan = {
        "selected_pages": [
            {"page": p.get("page"), "score": p.get("score"), "matched_terms": p.get("matched_terms")}
            for p in plan.get("selected_pages", [])
        ],
        "validated_document_terms": plan.get("validated_document_terms", [])[:40],
    }
    system = (
        "You synthesize a paper card from selected evidence pages. Return valid JSON only. "
        "Separate what the paper actually shows from how it may transfer to TechTrack."
    )
    user = (
        f"Paper: {paper['short_path']}\n"
        f"Module: {paper['module_id']} {paper['module_name']}\n\n"
        f"Project context:\n{case_context}\n\n"
        "Return JSON with required keys: paper_actual_contribution, evidence_used, techtrack_upgrade_role, "
        "capacity_allocation_role, assurance_test_role, what_not_to_claim, study_notes, citation_anchors, "
        "presentation_questions. Keep it specific and page-anchored.\n\n"
        f"Retrieval plan:\n{json.dumps(compact_plan, indent=2)}\n\n"
        f"Selected-page card:\n{json.dumps(page_card, indent=2)}\n\n"
        f"Attempt: {attempt}"
    )
    raw = llm.complete(system, user, temperature=0.1, max_tokens=cfg.paper_card_max_tokens * attempt)
    obj = _parse_json(raw, defaults={})
    obj.setdefault("short_path", paper["short_path"])
    obj.setdefault("module_id", paper["module_id"])
    obj.setdefault("module_name", paper["module_name"])
    obj.setdefault("filename", paper["filename"])
    obj.setdefault("selected_pages", [p.get("page") for p in plan.get("selected_pages", [])])
    obj["raw_model_output"] = obj.get("raw_model_output", raw)
    return obj


def _synthesize_network(llm, paper_cards: list[dict], case_context: str, cfg: SiraSweepConfig) -> str:
    compact = []
    for card in paper_cards:
        compact.append({
            "short_path": card.get("short_path"),
            "module": f"{card.get('module_id')} {card.get('module_name')}",
            "actual": card.get("paper_actual_contribution"),
            "techtrack": card.get("techtrack_upgrade_role"),
            "capacity": card.get("capacity_allocation_role"),
            "tests": card.get("assurance_test_role"),
            "limits": card.get("what_not_to_claim"),
            "anchors": card.get("citation_anchors"),
        })
    system = (
        "You synthesize a SIRA-selected evidence network for an Assured Autonomy project. "
        "Use the cited paper cards substantively and avoid generic summaries."
    )
    user = (
        f"Project context:\n{case_context}\n\n"
        f"Paper cards:\n{json.dumps(compact, indent=2)}\n\n"
        "Produce a concise but rich synthesis with: "
        "1. which evidence is strong, 2. which evidence is weak or analogical, "
        "3. TechTrack upgrade design implications, 4. capacity-allocation hypotheses, "
        "5. cost-aware assurance tests, 6. paper-presentation candidates, "
        "7. next deep-search questions. Cite compact paper paths and page anchors."
    )
    return llm.complete(system, user, temperature=0.1, max_tokens=cfg.synthesis_max_tokens)


def _paper_sample(pages: list[dict], max_chars: int) -> str:
    if not pages:
        return ""
    selected = []
    front = pages[:3]
    middle = pages[len(pages) // 2: len(pages) // 2 + 2]
    back = pages[-2:] if len(pages) > 5 else []
    for page in front + middle + back:
        selected.append(f"[Page {page['page']}]\n{page['text']}")
    text = "\n\n".join(selected)
    return text[:max_chars]


def _matched_terms(text: str, weighted_terms: list[tuple[str, float, dict]]) -> list[str]:
    lower = normalize_space(text).lower()
    out = []
    for term, _, _ in weighted_terms:
        if term in lower:
            out.append(term)
        else:
            pieces = _tokens(term)
            if pieces and all(piece in lower for piece in pieces):
                out.append(term)
    return list(dict.fromkeys(out))


def _snippet(text: str, matched: list[str], limit: int) -> str:
    if not text:
        return ""
    lower = text.lower()
    positions = [lower.find(term) for term in matched if lower.find(term) >= 0]
    pos = min(positions) if positions else 0
    start = max(0, pos - limit // 3)
    end = min(len(text), start + limit)
    return normalize_space(text[start:end])


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


def _parse_json(raw: str, defaults: dict | None = None) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return {**(defaults or {}), **value}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw or "", flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return {**(defaults or {}), **value}
        except Exception:
            pass
    return {**(defaults or {}), "status": "parse_failed", "raw_model_output": raw or "", "validation_error": "not valid JSON"}


def _render_paper_card_md(card: dict) -> str:
    return "\n".join([
        f"# {card.get('short_path')}",
        "",
        f"Module: {card.get('module_id')} {card.get('module_name')}",
        "",
        "## What The Paper Actually Contributes",
        str(card.get("paper_actual_contribution", "")),
        "",
        "## TechTrack Upgrade Role",
        str(card.get("techtrack_upgrade_role", "")),
        "",
        "## Capacity Allocation Role",
        str(card.get("capacity_allocation_role", "")),
        "",
        "## Assurance Test Role",
        str(card.get("assurance_test_role", "")),
        "",
        "## What Not To Claim",
        json.dumps(card.get("what_not_to_claim", []), indent=2),
        "",
        "## Citation Anchors",
        json.dumps(card.get("citation_anchors", []), indent=2),
        "",
    ])


def _load_cards(path: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]


def _case_context(cfg: SiraSweepConfig) -> str:
    if cfg.case_file:
        return cfg.case_file.read_text(encoding="utf-8")
    if cfg.case_text:
        return cfg.case_text
    return DEFAULT_CASE


def _clean(text: str) -> str:
    return normalize_space(text or "")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (text or "").lower())
        if t not in STOPWORDS
    ]


def _normalize_term(term: str) -> str:
    term = normalize_space(str(term or "")).lower()
    term = re.sub(r"^[^\w]+|[^\w]+$", "", term)
    if len(term) < 3 or len(term) > 80:
        return ""
    if term in STOPWORDS:
        return ""
    return term


def _new_run_dir(base: Path, model: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return base / f"sira_sweep_{_safe_stem(model)}_{stamp}"


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "item"


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "been",
    "have", "has", "had", "not", "but", "can", "will", "would", "could", "should",
    "into", "between", "their", "there", "than", "then", "these", "those", "also",
    "such", "using", "used", "use", "our", "all", "any", "more", "most", "paper",
    "system", "systems", "model", "models", "data", "learning", "approach", "method",
    "methods", "results", "based", "show", "shows", "shown", "may", "figure", "table",
    "pdf",
}


SEED_QUERY_TERMS = [
    "capacity allocation",
    "spectral norm",
    "stable rank",
    "effective rank",
    "hard negative mining",
    "dataset shift",
    "covariate shift",
    "out-of-distribution",
    "adversarial examples",
    "evasion attack",
    "object detection",
    "traffic sign",
    "runtime assurance",
    "formal verification",
    "calibration",
    "false negative",
    "small object",
    "robustness",
    "fault detection",
    "anomaly detection",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", action="append",
                        help="Directory to scan. Repeat for course and external paper roots.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--out-dir", default="sira_sweep_runs")
    parser.add_argument("--resume-dir")
    parser.add_argument("--case-file")
    parser.add_argument("--case-text")
    parser.add_argument("--modules", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pages-per-pdf", type=int, default=500)
    parser.add_argument("--sample-chars", type=int, default=18000)
    parser.add_argument("--top-pages-per-paper", type=int, default=8)
    parser.add_argument("--max-selected-page-chars", type=int, default=18000)
    parser.add_argument("--max-df-ratio", type=float, default=0.45)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--query-weight", type=float, default=3.0)
    parser.add_argument("--document-weight", type=float, default=0.85)
    parser.add_argument("--vocab-max-tokens", type=int, default=1600)
    parser.add_argument("--page-card-max-tokens", type=int, default=2600)
    parser.add_argument("--paper-card-max-tokens", type=int, default=4200)
    parser.add_argument("--synthesis-max-tokens", type=int, default=9000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--synthesize-only", action="store_true")
    args = parser.parse_args(argv)

    roots = [Path(p) for p in (args.papers_dir or [])] or [
        Path('papers/papers'),
        Path('papers/external'),
    ]
    run_sira_sweep(SiraSweepConfig(
        papers_dirs=roots,
        provider=args.provider,
        model=args.model,
        out_dir=Path(args.out_dir),
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
        case_file=Path(args.case_file) if args.case_file else None,
        case_text=args.case_text,
        modules=args.modules,
        limit=args.limit,
        max_pages_per_pdf=args.max_pages_per_pdf,
        sample_chars=args.sample_chars,
        top_pages_per_paper=args.top_pages_per_paper,
        max_selected_page_chars=args.max_selected_page_chars,
        max_df_ratio=args.max_df_ratio,
        min_df=args.min_df,
        query_weight=args.query_weight,
        document_weight=args.document_weight,
        vocab_max_tokens=args.vocab_max_tokens,
        page_card_max_tokens=args.page_card_max_tokens,
        paper_card_max_tokens=args.paper_card_max_tokens,
        synthesis_max_tokens=args.synthesis_max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
        retrieval_only=args.retrieval_only,
        synthesize_only=args.synthesize_only,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic instruction dataset generation for PEFT domain alignment."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .config import AA_MODULES, module_id_from_dirname
from .ingest import Chunk, ingest_course_corpus
from .paths import DEFAULT_DATASET_PATH, PAPERS_DIR, setup_paths
from .pdf_text import extract_pdf_pages


def _abstract(text: str, limit: int = 1800) -> str:
    m = re.search(r"\babstract\b[:\s]*(.*?)(?:\n\s*(?:1\.|introduction\b|keywords\b))",
                  text, flags=re.IGNORECASE | re.DOTALL)
    value = m.group(1) if m else text[:limit]
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _concrete_problem(text: str, limit: int = 1800) -> str:
    patterns = (
        r"concrete problems?.{0,80}?(.*?)(?:\n\s*(?:\d+\.|[A-Z][A-Za-z ]{3,}:))",
        r"problem formulation.{0,80}?(.*?)(?:\n\s*(?:\d+\.|[A-Z][A-Za-z ]{3,}:))",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:limit]
    return ""


def _author_year(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return stem


def _citation(path: Path, module_name: str) -> str:
    return f"| Source: {_author_year(path.name)} | Module: {module_name} | Local Path: {path}"


def _chunk_citation(chunk: Chunk, papers_root: Path) -> str:
    local_path = papers_root / chunk.relpath
    return (
        f"| Source: {_author_year(chunk.filename)} | Module: {chunk.module_name} | "
        f"Local Path: {local_path}"
    )


def _brief(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "..."


def examples_for_pdf(path: Path, papers_root: Path) -> list[dict]:
    module_id = None
    for parent in path.parents:
        module_id = module_id_from_dirname(parent.name)
        if module_id:
            break
    if not module_id:
        return []
    pages = extract_pdf_pages(path, max_pages=8)
    head = "\n".join(pages)
    abstract = _abstract(head)
    concrete = _concrete_problem(head)
    module_name = AA_MODULES[module_id]
    local_path = str(path)
    source = _author_year(path.name)
    cite = _citation(path, module_name)
    rows = [
        {
            "instruction": "Identify the Assured Autonomy module for this required reading and justify it.",
            "input": f"Reading: {path.name}\nAbstract: {abstract}",
            "output": (
                f"{path.name} belongs to {module_id}: {module_name}. "
                f"The abstract emphasizes concepts aligned with {module_name}. " +
                cite
            ),
            "module_id": module_id,
            "module_name": module_name,
            "source_path": local_path,
        },
        {
            "instruction": "Answer in the required citation format using the local course reading.",
            "input": f"What is the relevance of this paper to assured autonomy?\n{abstract}",
            "output": (
                f"This reading provides module-specific evidence for {module_name}. " +
                cite
            ),
            "module_id": module_id,
            "module_name": module_name,
            "source_path": local_path,
        },
    ]
    if concrete:
        rows.append({
            "instruction": "Extract the concrete problem statement and connect it to the course module.",
            "input": f"Reading: {path.name}\nConcrete problem passage: {concrete}",
            "output": (
                f"The concrete problem concerns risks handled in {module_id}: {module_name}. " +
                cite
            ),
            "module_id": module_id,
            "module_name": module_name,
            "source_path": local_path,
        })
    return rows


def examples_for_chunk(chunk: Chunk, papers_root: Path) -> list[dict]:
    passage = _brief(chunk.text)
    cite = _chunk_citation(chunk, papers_root)
    return [
        {
            "instruction": "Summarize this course passage in one concise answer sentence with the required inline citation format.",
            "input": f"Passage from {chunk.module_name}:\n{passage}",
            "output": f"This passage supports {chunk.module_name} by describing a course-relevant method, risk, or design concern. {cite}",
            "module_id": chunk.module_id,
            "module_name": chunk.module_name,
            "source_path": str(papers_root / chunk.relpath),
        },
        {
            "instruction": "Classify the passage into the correct Assured Autonomy module and cite the local paper.",
            "input": passage,
            "output": f"The passage belongs to {chunk.module_id}: {chunk.module_name}. {cite}",
            "module_id": chunk.module_id,
            "module_name": chunk.module_name,
            "source_path": str(papers_root / chunk.relpath),
        },
    ]


COMPARISON_CASES: tuple[tuple[str, str, str], ...] = (
    ("M12", "M06", "Compare Runtime Assurance and Formal Verification."),
    ("M05", "M04", "Compare Dataset Shift and Anomaly Detection for autonomous systems."),
    ("M12", "M13", "How does Runtime Assurance complement Safe Reinforcement Learning?"),
    ("M11", "M03", "How does Adversarial Autonomy differ from Safety, Security, and Privacy?"),
    ("M08", "M09", "Compare Interpretable Autonomy and Human-AI Integration."),
)


def examples_for_comparisons(chunks: list[Chunk], papers_root: Path,
                             per_pair: int = 12) -> list[dict]:
    by_module: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_module[chunk.module_id].append(chunk)
    rows: list[dict] = []
    for left_id, right_id, question in COMPARISON_CASES:
        left = by_module.get(left_id, [])[:per_pair]
        right = by_module.get(right_id, [])[:per_pair]
        for a, b in zip(left, right):
            cite_a = _chunk_citation(a, papers_root)
            cite_b = _chunk_citation(b, papers_root)
            rows.append({
                "instruction": "Answer the comparison question. Use exactly two concise paragraphs and cite one source from each module using the required inline citation format. Do not create a References section.",
                "input": (
                    f"Question: {question}\n\n"
                    f"Passage A ({a.module_name}): {_brief(a.text, 500)}\n\n"
                    f"Passage B ({b.module_name}): {_brief(b.text, 500)}"
                ),
                "output": (
                    f"{a.module_name} contributes the first side of the comparison by grounding the answer in its module-specific concepts. {cite_a}\n\n"
                    f"{b.module_name} contributes the second side of the comparison, so the final answer should explicitly distinguish it rather than merge the modules. {cite_b}"
                ),
                "module_id": f"{left_id}+{right_id}",
                "module_name": f"{AA_MODULES[left_id]} / {AA_MODULES[right_id]}",
                "source_path": f"{papers_root / a.relpath}; {papers_root / b.relpath}",
            })
    return rows


def build_instruction_dataset(papers_dir: str | Path = PAPERS_DIR,
                              out_path: str | Path = DEFAULT_DATASET_PATH,
                              expanded: bool = False,
                              max_chunks_per_module: int = 60) -> int:
    papers_root = Path(papers_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for pdf in sorted(papers_root.rglob("*.pdf")):
            if "optional" in str(pdf).lower():
                continue
            for row in examples_for_pdf(pdf, papers_root):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        if expanded:
            chunks = ingest_course_corpus(papers_root, verbose=False)
            by_module: dict[str, list[Chunk]] = defaultdict(list)
            for chunk in chunks:
                if "optional" in chunk.relpath.lower():
                    continue
                by_module[chunk.module_id].append(chunk)
            selected: list[Chunk] = []
            for module_id in sorted(by_module):
                selected.extend(by_module[module_id][:max_chunks_per_module])
            for chunk in selected:
                for row in examples_for_chunk(chunk, papers_root):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            for row in examples_for_comparisons(selected, papers_root):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    paths = setup_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", default=str(paths.papers))
    parser.add_argument("--out", default=str(paths.dataset_path))
    parser.add_argument("--expanded", action="store_true",
                        help="Add chunk-level and cross-module task-alignment examples.")
    parser.add_argument("--max-chunks-per-module", type=int, default=60)
    args = parser.parse_args()
    count = build_instruction_dataset(
        args.papers_dir,
        args.out,
        expanded=args.expanded,
        max_chunks_per_module=args.max_chunks_per_module,
    )
    print(f"Wrote {count} instruction examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

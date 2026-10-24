"""Public package facade for the Assured Autonomy RAG toolkit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_root_module(name: str):
    fqname = f"{__name__}.{name}"
    if fqname in sys.modules:
        return sys.modules[fqname]
    path = _ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(fqname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {fqname} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = module
    spec.loader.exec_module(module)
    return module


for _name in (
    "config", "ingest", "loaders", "indexing", "llm", "rubric",
    "analyze", "compare", "paths", "pdf_text", "syllabus",
    "hybrid_retriever", "prompt_router", "agentic_rag", "fine_tuning_dataset",
    "fine_tune_peft", "evaluate_agent", "iterative_rag", "policy_rag",
    "policy_debate", "concept_debate", "corpus_recommender", "case_concept_loop",
    "data_access_batch", "semantic_sweep", "techtrack_synthesis", "external_papers",
    "deep_pdf_search", "full_reading_sweep", "sira_sweep", "sira_card_curator",
    "card_curator", "bridge_graph_rag", "concept_graph", "council_bridge_rag",
    "code_reading_upgrade", "assurance_case_cards", "reading_map",
    "rapid_classifier", "corpus_recsys", "cli",
):
    if (_ROOT / f"{_name}.py").exists():
        _load_root_module(_name)

from .agentic_rag import AgenticRagAgent, AgenticRagConfig
from .analyze import analyze_baseline, analyze_with_rag, analyze_with_rubric
from .compare import compare_modes
from .fine_tuning_dataset import build_instruction_dataset
from .hybrid_retriever import HybridPaperRetriever
from .indexing import RagIndex, build_index
from .ingest import Chunk, ingest_course_corpus
from .llm import get_llm
from .loaders import load_new_item
from .syllabus import SyllabusNavigator

__all__ = [
    "AgenticRagAgent", "AgenticRagConfig", "Chunk", "HybridPaperRetriever",
    "RagIndex", "SyllabusNavigator", "analyze_baseline", "analyze_with_rag",
    "analyze_with_rubric", "build_index", "build_instruction_dataset",
    "get_llm", "ingest_course_corpus", "load_new_item", "compare_modes",
]

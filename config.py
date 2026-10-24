"""Configuration and the JHU Assured Autonomy module taxonomy."""
from __future__ import annotations
from dataclasses import dataclass

# Canonical module IDs and human-readable names. Module IDs are the prefix
# of the per-module subdirectories under papers/ (e.g., "M03").
AA_MODULES: dict[str, str] = {
    "M01-02": "Introduction to Assured Autonomy",
    "M03": "Safety, Security, and Privacy",
    "M04": "Anomaly and Fault Detection",
    "M05": "Dataset Shift",
    "M06": "Verification and Validation",
    "M07": "Test and Evaluation",
    "M08": "Interpretable Autonomy",
    "M09": "Human-AI Integration",
    "M10": "Policy and Governance",
    "M11": "Adversarial Autonomy",
    "M12": "Runtime Assurance",
    "M13": "Safe Reinforcement Learning",
    "M14": "Multi-Agent and Cooperative Systems",
}


@dataclass
class IngestConfig:
    chunk_chars: int = 1400        # ~ 300-350 tokens
    chunk_overlap_chars: int = 200
    min_chunk_chars: int = 200     # drop tiny fragments
    max_pages_per_doc: int = 200   # safety bound


@dataclass
class IndexConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384       # MiniLM-L6
    normalize: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 8
    min_score: float = 0.20        # cosine similarity floor


def module_id_from_dirname(dirname: str) -> str | None:
    """Extract canonical module id from a directory name like 'M03_SafetySecurityPrivacy'.

    Returns None if no match.
    """
    if not dirname:
        return None
    head = dirname.split("_", 1)[0]
    # Handle the M01-02 combined module
    if head in AA_MODULES:
        return head
    # Try a fuzzier match (some folks may name dirs with just M3 etc.)
    for k in AA_MODULES:
        if dirname.startswith(k + "_") or dirname == k:
            return k
    return None

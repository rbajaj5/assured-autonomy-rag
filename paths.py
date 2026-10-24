"""Canonical local paths for the JHU Assured Autonomy RAG project."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AATE_PROJECT_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
PAPERS_DIR = PROJECT_ROOT / "papers"
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".aate_index"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "aa_instruction_tuning.jsonl"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "models" / "aa_lora_adapter"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    papers: Path = PAPERS_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    data_dir: Path = DEFAULT_DATA_DIR
    dataset_path: Path = DEFAULT_DATASET_PATH
    adapter_dir: Path = DEFAULT_ADAPTER_DIR


def setup_paths() -> ProjectPaths:
    """Validate and create local project directories used by the framework."""
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root not found: {PROJECT_ROOT}")
    if not PAPERS_DIR.exists():
        raise FileNotFoundError(f"Papers directory not found: {PAPERS_DIR}")
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    return ProjectPaths()

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from rllm.data import Dataset, DatasetRegistry

logger = logging.getLogger(__name__)

# In-memory cache: (name, split) -> list[dict]
# Populated by register_jsonl/register_puzzles so train.py can call
# get_dataset() instead of DatasetRegistry.load_dataset() (which does a
# slow parquet → polars → Python-dict round-trip: ~254s for 500k rows).
_DATASET_CACHE: dict[tuple[str, str], list[dict]] = {}


def register_jsonl(path: str | Path, name: str, split: str) -> int:
    """Register a JSONL file as an rllm dataset split.

    Uses HuggingFace Arrow parsing (fast) and caches rows in memory so
    get_dataset() can return them without a parquet round-trip.
    Skips writing the parquet cache if it is already newer than the source.

    Args:
        path (str | Path): Path to the JSONL file.
        name (str): Dataset name to register under.
        split (str): Split name (e.g. ``"train"``, ``"test"``).

    Returns:
        int: Number of examples registered.
    """
    path = Path(path)

    # Check whether the on-disk parquet cache is still fresh.
    registry = DatasetRegistry._load_registry()
    split_info = registry.get("datasets", {}).get(name, {}).get("splits", {}).get(split)
    if split_info:
        parquet_path = Path(DatasetRegistry._DATASET_DIR) / split_info["path"]
        if parquet_path.exists() and parquet_path.stat().st_mtime >= path.stat().st_mtime:
            n = split_info["num_examples"]
            logger.info(f"Dataset '{name}/{split}' parquet cache fresh ({n} examples).")
            # Load into memory via Arrow (fast) rather than polars parquet read.
            ds = hf_load_dataset("json", data_files=str(path), split="train")
            rows = ds.to_list()
            _DATASET_CACHE[(name, split)] = rows
            return n

    # First run or source changed: parse, write parquet, cache in memory.
    ds = hf_load_dataset("json", data_files=str(path), split="train")
    rows = ds.to_list()
    _DATASET_CACHE[(name, split)] = rows
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)


def get_dataset(name: str, split: str) -> Dataset:
    """Return the in-memory dataset populated by register_jsonl.

    Avoids the ~254s DatasetRegistry.load_dataset parquet round-trip.

    Args:
        name (str): Dataset name.
        split (str): Split name.

    Returns:
        Dataset: rllm Dataset object.
    """
    rows = _DATASET_CACHE.get((name, split))
    if rows is None:
        raise KeyError(f"Dataset '{name}/{split}' not in cache — call register_jsonl first.")
    return Dataset(data=rows, name=name, split=split)


def register_puzzles(path: str | Path, name: str, split: str = "test") -> int:
    """Register a chess_llm-style puzzles JSON file as an rllm dataset split.

    Args:
        path (str | Path): Path to the puzzles JSON file.
        name (str): Dataset name to register under.
        split (str): Split name.

    Returns:
        int: Number of examples registered.
    """
    raw: list[list] = json.loads(Path(path).read_text())
    rows = [{"fen": fen, "move": best_move, "legal_moves": legal} for fen, best_move, legal in raw]
    _DATASET_CACHE[(name, split)] = rows
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)

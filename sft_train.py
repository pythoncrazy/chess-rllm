"""SFT training on Stockfish-generated chess positions.

Usage:
    uv run python sft_train.py \\
        train_files=data/sft_train.jsonl \\
        val_files=data/sft_val.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

logger = logging.getLogger(__name__)


def load_jsonl_as_dataset(path: str) -> Dataset:
    """Load a JSONL file with 'messages' rows into a HuggingFace Dataset."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


@hydra.main(config_path="conf", config_name="sft", version_base=None)
def main(config: DictConfig) -> None:
    train_files = config.data.get("train_files")
    val_files = config.data.get("val_files", None)

    if not train_files:
        raise ValueError("data.train_files must be set (path to JSONL)")

    train_dataset = load_jsonl_as_dataset(train_files)
    val_dataset = load_jsonl_as_dataset(val_files) if val_files else None

    AgentSFTTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="tinker",
    ).train()


if __name__ == "__main__":
    main()

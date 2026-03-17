"""SFT training on Stockfish-generated chess positions.

Usage:
    uv run python sft_train.py \\
        data.train_files=data/sf_train.jsonl \\
        data.val_files=data/sf_val.jsonl
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import hydra
import tinker
from datasets import load_dataset
from omegaconf import DictConfig
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from tinker_cookbook.supervised.common import datum_from_model_input_weights

# TinkerSFTDataset lives in the deprecated module — we patch get_batch so that
# the actual tokenisation no longer imports anything else from there.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from rllm.trainer.deprecated.tinker_sft_dataset import TinkerSFTDataset

logger = logging.getLogger(__name__)

_TOKENIZER_POOL = ThreadPoolExecutor(max_workers=32)


def _fast_get_batch(self, index: int) -> list[tinker.Datum]:
    """Parallel drop-in replacement for TinkerSFTDataset.get_batch.

    Fixes two bottlenecks in the original:
    1. Replaces 512 individual self.dataset[i] calls with one Arrow batch slice.
    2. Parallelises tokenisation across all rows with threads — the Rust-backed
       Qwen3 tokeniser releases the GIL, so threads give real speedup.
    """
    start_idx = index * self.batch_size
    end_idx = min(start_idx + self.batch_size, len(self.dataset))
    batch = self.dataset[start_idx:end_idx]  # single vectorised Arrow slice
    messages_list = batch["messages"]

    def _tok(messages):
        model_input, weights = self.renderer.build_supervised_example(
            messages, train_on_what=self.train_on_what
        )
        return datum_from_model_input_weights(model_input, weights, self.max_length)

    return list(_TOKENIZER_POOL.map(_tok, messages_list))


TinkerSFTDataset.get_batch = _fast_get_batch


def load_jsonl_as_dataset(path: str):
    """Load a JSONL file with 'messages' rows into a HuggingFace Dataset."""
    ds = load_dataset("json", data_files=path, split="train")
    logger.info(f"Loaded {len(ds)} examples from {path}")
    return ds


@hydra.main(config_path="conf", config_name="sft", version_base=None)
def main(config: DictConfig) -> None:
    train_files = config.data.get("train_files")
    val_files = config.data.get("val_files", None)

    if not train_files:
        raise ValueError("data.train_files must be set (path to JSONL)")

    train_dataset = load_jsonl_as_dataset(train_files)
    val_dataset = load_jsonl_as_dataset(val_files) if val_files else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        AgentSFTTrainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            backend="tinker",
        ).train()


if __name__ == "__main__":
    main()

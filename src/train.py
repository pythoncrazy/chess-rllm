from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rllm.experimental.unified_trainer import AgentTrainer

from data import get_dataset, register_jsonl, register_puzzles
from eval import ChessWorkflow, chess_reward_fn


def _setup_rollout_file_logging(log_dir: str) -> None:
    """Redirect eval (rollout) logs to a file instead of console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "rollouts.log")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    eval_logger = logging.getLogger("eval")
    eval_logger.addHandler(handler)
    eval_logger.propagate = False
    logging.getLogger(__name__).info(f"Rollout logs -> {log_path}")


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    """Train a chess RL agent with GRPO from pre-generated JSONL datasets.

    Pass Hydra overrides on the CLI, e.g.::

        python train.py run_name=grpo-v1 train_files=data/sf_train_grpo.jsonl val_files=data/sf_val.jsonl

    Args:
        config (DictConfig): Hydra config merged from ``train.yaml``.
            Required keys: ``train_files`` (JSONL path), ``val_files`` or ``val_puzzles``.
    """
    train_files: str = config.get("train_files")
    val_files: str | None = config.get("val_files", None)
    val_puzzles: str | None = config.get("val_puzzles", None)

    if not train_files:
        raise ValueError("train_files must be set (path to JSONL)")

    run_name = config.get("run_name", "unnamed")
    log_dir = config.get("log_dir", f"outputs/rollouts/{run_name}")
    _setup_rollout_file_logging(log_dir)

    register_jsonl(train_files, "chess", "train")

    if val_files:
        register_jsonl(val_files, "chess", "test")
    elif val_puzzles:
        register_puzzles(val_puzzles, "chess", "test")
    else:
        raise ValueError("val_files or val_puzzles must be set")

    AgentTrainer(
        workflow_class=ChessWorkflow,
        workflow_args={"reward_function": chess_reward_fn},
        config=config,
        train_dataset=get_dataset("chess", "train"),
        val_dataset=get_dataset("chess", "test"),
        backend="tinker",
    ).train()


if __name__ == "__main__":
    main()

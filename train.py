from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

from data import register_lichess_games, register_puzzles
from eval import ChessWorkflow, chess_reward_fn


def _setup_rollout_file_logging(log_dir: str) -> None:
    """Redirect eval (rollout) logs to a file instead of console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "rollouts.log")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    eval_logger = logging.getLogger("eval")
    eval_logger.addHandler(handler)
    eval_logger.propagate = False  # don't also print to console
    logging.getLogger(__name__).info(f"Rollout logs -> {log_path}")


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    """Train a chess RL agent with GRPO, streaming positions from Lichess/standard-chess-games.

    Pass Hydra overrides on the CLI, e.g.::

        python train.py rllm/backend=tinker +max_train=10000

    Args:
        config (DictConfig): Hydra config merged from ``unified.yaml`` + ``rllm/backend/tinker.yaml``.
            Extra keys: ``max_train`` (int, default 50000), ``max_val`` (int, default 1000),
            ``val_puzzles`` (str path to chess_llm puzzles.json, optional),
            ``sample_every`` (int, default 5), ``min_ply`` (int, default 20),
            ``max_ply`` (int, default 80).
    """
    max_train: int = config.get("max_train", 50_000)
    max_val: int = config.get("max_val", 1_000)
    val_puzzles: str | None = config.get("val_puzzles", None)
    sample_every: int = config.get("sample_every", 5)
    min_ply: int = config.get("min_ply", 20)
    max_ply: int = config.get("max_ply", 80)

    OmegaConf.update(config, "rllm.trainer.logger", list(config.rllm.trainer.logger) + ["ui"], merge=True)

    log_dir = config.get("log_dir", "outputs/rollouts")
    _setup_rollout_file_logging(log_dir)

    register_lichess_games("chess", "train", max_train, sample_every, min_ply, max_ply)
    if val_puzzles:
        register_puzzles(val_puzzles, "chess", "test")
    else:
        register_lichess_games("chess", "test", max_val, sample_every, min_ply, max_ply)

    AgentTrainer(
        workflow_class=ChessWorkflow,
        workflow_args={"reward_function": chess_reward_fn},
        config=config,
        train_dataset=DatasetRegistry.load_dataset("chess", "train"),
        val_dataset=DatasetRegistry.load_dataset("chess", "test"),
        backend="tinker",
    ).train()


if __name__ == "__main__":
    main()

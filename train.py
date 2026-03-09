from __future__ import annotations

import click
from omegaconf import OmegaConf
from rllm.data import DatasetRegistry
from rllm.experimental.cli.train import build_train_config, make_agent_run_func
from rllm.experimental.unified_trainer import AgentTrainer

from data import register_jsonl, register_puzzles
from eval import ChessAgent, ChessEvaluator


@click.command()
@click.option("--model", default="Qwen/Qwen3-8B", show_default=True, help="Model name or HuggingFace path.")
@click.option("--dataset", default="chess", show_default=True, help="Registered dataset name.")
@click.option("--train-data", default=None, type=click.Path(exists=True), help="JSONL file to register as train split.")
@click.option("--val-data", default=None, type=click.Path(exists=True), help="JSONL or puzzles JSON to register as test split.")
@click.option("--puzzles", is_flag=True, default=False, help="Treat --val-data as a chess_llm puzzles.json file.")
@click.option("--group-size", default=8, show_default=True, type=int, help="GRPO rollouts per prompt.")
@click.option("--batch-size", default=16, show_default=True, type=int, help="Training batch size.")
@click.option("--lr", default=2e-5, show_default=True, type=float, help="Learning rate.")
@click.option("--lora-rank", default=32, show_default=True, type=int, help="LoRA rank.")
@click.option("--epochs", default=1, show_default=True, type=int, help="Training epochs.")
@click.option("--max-steps", default=None, type=int, help="Stop after N steps (overrides --epochs).")
@click.option("--val-freq", default=5, show_default=True, type=int, help="Validate every N steps.")
@click.option("--save-freq", default=20, show_default=True, type=int, help="Checkpoint every N steps.")
@click.option("--project", default="chess-rllm", show_default=True, help="Project name for logging.")
@click.option("--experiment", default=None, help="Experiment name (defaults to --dataset).")
@click.option("--output", "output_dir", default="./checkpoints", show_default=True, type=click.Path(), help="Checkpoint directory.")
@click.option("--max-response-length", default=8192, show_default=True, type=int, help="Max response tokens (increase for thinking models).")
@click.option("--config", "config_file", default=None, type=click.Path(exists=True), help="YAML config merged over base templates.")
def train(
    model: str,
    dataset: str,
    train_data: str | None,
    val_data: str | None,
    puzzles: bool,
    group_size: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    epochs: int,
    max_steps: int | None,
    val_freq: int,
    save_freq: int,
    project: str,
    experiment: str | None,
    output_dir: str,
    max_response_length: int,
    config_file: str | None,
) -> None:
    """Train a chess RL agent with GRPO via the Tinker backend."""
    if train_data:
        register_jsonl(train_data, dataset, "train")
    if val_data:
        register_puzzles(val_data, dataset, "test") if puzzles else register_jsonl(val_data, dataset, "test")

    train_ds = DatasetRegistry.load_dataset(dataset, "train")
    val_ds = DatasetRegistry.load_dataset(dataset, "test") if val_data else None

    agent = ChessAgent()
    evaluator = ChessEvaluator()

    config = build_train_config(
        model_name=model,
        group_size=group_size,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        total_epochs=epochs,
        total_steps=max_steps,
        val_freq=val_freq,
        save_freq=save_freq,
        project=project,
        experiment=experiment or dataset,
        output_dir=output_dir,
        config_file=config_file,
    )
    config = OmegaConf.merge(config, OmegaConf.create({
        "data": {"max_response_length": max_response_length},
    }))

    AgentTrainer(
        backend="tinker",
        agent_run_func=make_agent_run_func(agent, evaluator, model),
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
    ).train()


if __name__ == "__main__":
    train()

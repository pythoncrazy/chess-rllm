from __future__ import annotations

import logging
import re
from collections import defaultdict

import chess
import numpy as np
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.simple_workflow import SimpleWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_USER_PROMPT_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_PROMPT_SUFFIX = "\n\nGive your answer in <answer>...</answer> tags."


def chess_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Evaluate a predicted chess move against the reference move in the task.

    Extracts the move from ``<answer>...</answer>`` tags, accepts both SAN and UCI
    notation, and falls back to binary reward when no Stockfish ``"moves"`` dict
    is present.

    Args:
        task_info (dict): Task row with ``"fen"``, ``"move"`` (SAN), and optionally
            ``"moves"`` (dict mapping SAN -> ``{"score": int}``).
        action (str): Raw model response string.

    Returns:
        RewardOutput: Reward in ``[0, 1]`` with ``is_correct`` flag.
    """
    if not isinstance(action, str):
        action = action.action
    m = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
    predicted = m.group(1).strip() if m else ""
    predicted = predicted.rstrip("!?")

    board = chess.Board(task_info["fen"])
    # Normalize SAN keys by stripping check/mate symbols for fuzzy matching
    san_map = {board.san(mv): mv for mv in board.legal_moves}
    san_map_norm = {s.rstrip("+#"): mv for s, mv in san_map.items()}
    uci_map = {mv.uci(): mv for mv in board.legal_moves}
    move = san_map.get(predicted) or san_map_norm.get(predicted) or uci_map.get(predicted)

    if move is None:
        logger.info("ILLEGAL  fen=%-72s predicted=%-10s response=%s", task_info["fen"], predicted, action[:200])
        return RewardOutput(reward=0.0, is_correct=False, metadata={"legal": False})

    best_str = task_info["move"]
    best_move = san_map.get(best_str) or uci_map.get(best_str)
    is_best = move == best_move

    moves_dict: dict | None = task_info.get("moves")
    if moves_dict and predicted in moves_dict:
        raw = moves_dict[predicted]
        score = int(raw["score"]) if isinstance(raw, dict) else int(raw)
        best_score = max(
            (int(v["score"]) if isinstance(v, dict) else int(v)) for v in moves_dict.values()
        )
        reward = max(0.0, min(1.0, score / best_score)) if best_score > 0 else 0.0
    else:
        # Small legality reward so GRPO has gradient signal even before any correct moves
        reward = 1.0 if is_best else 0.1

    logger.info("%-7s  fen=%-72s predicted=%-10s expected=%-10s reward=%.2f", "CORRECT" if is_best else "WRONG", task_info["fen"], predicted, best_str, reward)
    return RewardOutput(reward=reward, is_correct=is_best, metadata={"legal": True, "is_correct": is_best})


class ChessWorkflow(SimpleWorkflow):
    """Single-turn chess workflow that injects the system prompt and FEN as messages."""

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Run one chess turn: build messages from FEN and evaluate the response.

        Args:
            task (dict): Task row with at least a ``"fen"`` key.
            uid (str): Unique rollout identifier.

        Returns:
            Episode: Completed episode with reward.
        """
        self.reset(task, uid)
        messages = [{"role": "user", "content": _USER_PROMPT_PREFIX + task["fen"] + _USER_PROMPT_SUFFIX}]

        from rllm.engine import ModelOutput
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)
        logger.info("PROMPT   %s", messages[0]["content"])
        logger.info("RESPONSE %s", (output.content or "")[:300])
        action_str = output.content
        reward_result = self.reward_function({**task, "messages": messages}, action_str)

        trajectory = self.agent.trajectory
        trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                thought=output.reasoning,
                action=action_str,
                reward=reward_result.reward,
                model_output=output,
                metadata=reward_result.metadata,
            )
        )
        self.commit(agent=self.agent, reset=True)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
        raise TerminationEvent(TerminationReason.ENV_DONE)

    def collect_metrics(self, episode: Episode) -> None:
        """Collect reward and correctness metrics from the episode.

        Args:
            episode (Episode): Episode to collect metrics from.
        """
        rewards: list[float] = []
        correct: list[float] = []
        for traj in episode.trajectories:
            rewards.append(traj.reward)
            for step in traj.steps:
                if step.metadata:
                    correct.append(1.0 if step.metadata.get("is_correct") else 0.0)
        episode.metrics = {
            "default_traj_name_acc": float(np.mean(rewards)) if rewards else 0.0,
            "percent_correct": float(np.mean(correct)) if correct else 0.0,
            "num_correct": int(sum(correct)),
        }

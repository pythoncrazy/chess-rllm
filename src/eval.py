from __future__ import annotations

import logging
import math
import os
import queue
import re
import threading
from contextlib import contextmanager
from typing import Generator

import chess
import chess.engine
import numpy as np
from rllm.agents.agent import Episode, Step
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.simple_workflow import SimpleWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

_USER_PROMPT_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_PROMPT_SUFFIX = "\n\nGive your answer in SAN notation (e.g. Nf3) in <answer>...</answer> tags."

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_SF_POOL_SIZE = int(os.environ.get("SF_POOL_SIZE", "8"))
_SF_TIME = float(os.environ.get("SF_TIME", "0.05"))
_SF_LOSS_SCALE = float(os.environ.get("SF_LOSS_SCALE", "50"))
_MAX_CP = 10_000

_engine_pool: queue.Queue | None = None
_pool_lock = threading.Lock()


def _get_pool() -> queue.Queue:
    """Initialize and return the global Stockfish engine pool.

    Returns:
        queue.Queue: Thread-safe pool of ``chess.engine.SimpleEngine`` instances.
    """
    global _engine_pool
    if _engine_pool is None:
        with _pool_lock:
            if _engine_pool is None:
                pool: queue.Queue = queue.Queue()
                for _ in range(_SF_POOL_SIZE):
                    try:
                        pool.put(chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH))
                    except Exception as e:
                        logger.warning("Stockfish unavailable: %s", e)
                _engine_pool = pool
    return _engine_pool


@contextmanager
def _engine() -> Generator[chess.engine.SimpleEngine, None, None]:
    """Borrow a Stockfish engine from the pool, returning it on exit.

    Yields:
        chess.engine.SimpleEngine: A Stockfish engine instance.
    """
    pool = _get_pool()
    engine = pool.get()
    try:
        yield engine
    finally:
        pool.put(engine)


def _cp(score: chess.engine.Score) -> int:
    """Extract a centipawn value from a Stockfish score, capping mate scores.

    Args:
        score (chess.engine.Score): Stockfish score relative to the player to move.

    Returns:
        int: Centipawn value in [-_MAX_CP, _MAX_CP].
    """
    return _MAX_CP if score.is_mate() else (score.score() or 0)


def _sf_loss_reward(board: chess.Board, move: chess.Move) -> float:
    """Score a move as exp(-centipawn_loss / _SF_LOSS_SCALE) using live Stockfish.

    Evaluates the best achievable score from the current position, then the score
    after the predicted move, and converts the centipawn loss into [0, 1].

    Args:
        board (chess.Board): Current position (not modified).
        move (chess.Move): Legal move to evaluate.

    Returns:
        float: Reward in (0, 1] where 1.0 means best move and ~0 means a blunder.
    """
    with _engine() as engine:
        best_cp = _cp(engine.analyse(board, chess.engine.Limit(time=_SF_TIME))["score"].relative)
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return 1.0
        if board.is_game_over():
            board.pop()
            return 0.5
        pred_cp = -_cp(engine.analyse(board, chess.engine.Limit(time=_SF_TIME))["score"].relative)
        board.pop()
    return math.exp(-max(0, best_cp - pred_cp) / _SF_LOSS_SCALE)


def chess_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Evaluate a predicted chess move using Stockfish centipawn loss.

    Args:
        task_info (dict): Row with ``"fen"`` and ``"move"`` (best SAN).
        action (str): Raw model response containing ``<answer>move</answer>``.

    Returns:
        RewardOutput: Reward in [0, 1]; 0 for illegal moves.
    """
    if not isinstance(action, str):
        action = action.action
    m = re.search(r"<answer>(.*?)</answer>", action, re.DOTALL)
    predicted = m.group(1).strip().rstrip("!?") if m else ""

    board = chess.Board(task_info["fen"])
    san_map = {board.san(mv): mv for mv in board.legal_moves}
    san_map_norm = {s.rstrip("+#"): mv for s, mv in san_map.items()}
    uci_map = {mv.uci(): mv for mv in board.legal_moves}
    move = san_map.get(predicted) or san_map_norm.get(predicted) or uci_map.get(predicted)

    if move is None:
        logger.info("ILLEGAL  fen=%-72s predicted=%-10s", task_info["fen"], predicted)
        return RewardOutput(reward=0.0, is_correct=False, metadata={"legal": False})

    best_str = task_info["move"]
    best_move = san_map.get(best_str) or uci_map.get(best_str)
    is_best = move == best_move

    reward = _sf_loss_reward(board, move)

    logger.info(
        "%-7s  fen=%-72s predicted=%-10s expected=%-10s reward=%.3f",
        "CORRECT" if is_best else "WRONG", task_info["fen"], predicted, best_str, reward,
    )
    return RewardOutput(reward=reward, is_correct=is_best, metadata={"legal": True, "is_correct": is_best})


class ChessWorkflow(SimpleWorkflow):
    """Single-turn chess workflow that injects the system prompt and FEN as messages."""

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Run one chess turn and evaluate the response.

        Args:
            task (dict): Row with at least a ``"fen"`` key.
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
        content: str = output.content or ""
        reasoning: str = output.reasoning or ""
        reward_result = self.reward_function({**task, "messages": messages}, content)

        self.agent.trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": content, "reasoning": reasoning}],
                thought=reasoning,
                action=content,
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
            rewards.append(traj.reward or 0.0)
            for step in traj.steps:
                if step.metadata:
                    correct.append(1.0 if step.metadata.get("is_correct") else 0.0)
        episode.metrics = {
            "default_traj_name_acc": float(np.mean(rewards)) if rewards else 0.0,
            "percent_correct": float(np.mean(correct)) if correct else 0.0,
            "num_correct": int(sum(correct)),
        }

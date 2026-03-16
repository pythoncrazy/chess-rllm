from __future__ import annotations

import logging
import os
import queue
import re
import threading

import chess
import chess.engine
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

# ---------------------------------------------------------------------------
# Stockfish engine pool for live move quality evaluation
# ---------------------------------------------------------------------------
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_SF_POOL_SIZE = int(os.environ.get("SF_POOL_SIZE", "8"))
_SF_TIME = float(os.environ.get("SF_TIME", "0.05"))  # seconds per analysis
_MAX_CP = 10_000
_LEGAL_FLOOR = 0.4   # minimum reward for any Stockfish-ranked move
_POOR_MOVE_REWARD = 0.2  # reward for a legal move outside Stockfish's top-5

_engine_pool: queue.Queue | None = None
_pool_lock = threading.Lock()


def _get_pool() -> queue.Queue:
    global _engine_pool
    if _engine_pool is None:
        with _pool_lock:
            if _engine_pool is None:
                pool: queue.Queue = queue.Queue()
                for _ in range(_SF_POOL_SIZE):
                    try:
                        pool.put(chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH))
                    except Exception as e:
                        logger.warning("Failed to start Stockfish engine: %s", e)
                _engine_pool = pool
    return _engine_pool


def _sf_scores(fen: str) -> dict[str, int]:
    """Return {SAN: centipawns} for the top-5 moves via Stockfish.

    Returns an empty dict if Stockfish is unavailable or fails.
    Uses a thread-safe pool of engines so parallel rollouts don't block each other.
    """
    pool = _get_pool()
    engine = pool.get()
    try:
        board = chess.Board(fen)
        infos = engine.analyse(board, chess.engine.Limit(time=_SF_TIME), multipv=5)
        scores: dict[str, int] = {}
        for info in infos:
            if "pv" not in info or not info["pv"]:
                continue
            mv = info["pv"][0]
            san = board.san(mv)
            pov = info["score"].relative
            cp = _MAX_CP if pov.is_mate() else (pov.score() or 0)
            scores[san] = cp
        return scores
    except Exception as e:
        logger.debug("Stockfish analysis failed for %s: %s", fen, e)
        return {}
    finally:
        pool.put(engine)


def _quality_reward(predicted: str, scores: dict[str, int]) -> float:
    """Map a move's centipawn score to [_LEGAL_FLOOR, 1.0] via range normalization.

    If the move is not in the scored set (below top-5), returns _POOR_MOVE_REWARD.
    """
    if predicted not in scores:
        return _POOR_MOVE_REWARD
    best = max(scores.values())
    worst = min(scores.values())
    score_range = best - worst
    if score_range <= 0:
        return 1.0
    return _LEGAL_FLOOR + (1.0 - _LEGAL_FLOOR) * (scores[predicted] - worst) / score_range


def chess_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Evaluate a predicted chess move and assign a quality-based reward.

    Priority:
    1. Pre-computed Stockfish scores in ``task_info["moves"]`` (from SFT data).
    2. Live Stockfish evaluation via engine pool (covers any legal move).
    3. Binary fallback if Stockfish is unavailable.

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

    # 1. Pre-computed scores (SFT data has top-3 with centipawn values)
    moves_dict: dict | None = task_info.get("moves")
    if moves_dict:
        scores = {
            k: (int(v["score"]) if isinstance(v, dict) else int(v))
            for k, v in moves_dict.items()
        }
        reward = _quality_reward(predicted, scores)
        if predicted not in scores:
            # Move is legal but outside pre-computed top-3 — try live Stockfish
            live = _sf_scores(task_info["fen"])
            if live:
                reward = _quality_reward(predicted, live)

    # 2. No pre-computed data — use live Stockfish
    else:
        live = _sf_scores(task_info["fen"])
        if live:
            reward = _quality_reward(predicted, live)
        else:
            # 3. Stockfish unavailable — binary fallback
            reward = 1.0 if is_best else 0.1

    logger.info(
        "%-7s  fen=%-72s predicted=%-10s expected=%-10s reward=%.2f",
        "CORRECT" if is_best else "WRONG", task_info["fen"], predicted, best_str, reward,
    )
    return RewardOutput(reward=reward, is_correct=is_best, metadata={"legal": True, "is_correct": is_best})


class ChessWorkflow(SimpleWorkflow):
    """Single-turn chess workflow that injects the system prompt and FEN as messages."""

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
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

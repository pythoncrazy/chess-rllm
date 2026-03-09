from __future__ import annotations

import re

import chess
from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig, EvalOutput, Signal, _extract_agent_answer
from rllm.types import Episode, Step, Trajectory

_SYSTEM_PROMPT = (
    "You are a grandmaster chess player. "
    "Given a chess position in FEN notation, list your top candidate moves "
    "inside <think>...</think>, then give the best move in <answer>...</answer>. "
    "Keep thinking to a short comma-separated list of moves only — no explanations. "
    "Example: <think>Rxe7, Nf5, Qh5</think><answer>Rxe7</answer>"
)


class ChessAgent:
    """Agent that queries an LLM to predict the best move for a chess position."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        """Run the agent on a single chess task.

        Args:
            task (dict): Task dict with at least a ``"fen"`` key.
            config (AgentConfig): Runtime config providing ``base_url`` and ``model``.

        Returns:
            Episode: Episode with the predicted move in ``artifacts["answer"]``.
        """
        client = OpenAI(base_url=config.base_url, api_key="none")
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task["fen"] + "</board>"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = response.choices[0].message.content or ""
        m = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        answer = m.group(1).strip() if m else output.strip().split()[-1]
        return Episode(
            trajectories=[Trajectory(steps=[Step(output=output)], output=output)],
            artifacts={"answer": answer},
        )


class ChessEvaluator:
    """Evaluator for chess move quality using python-chess.

    Reward is proportional to move score when a ``"moves"`` dict is present in
    the task, otherwise binary (1.0 for best move, 0.0 otherwise). Illegal moves
    always receive 0.0.
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        """Evaluate a chess episode against the ground-truth best move.

        Args:
            task (dict): Task dict with ``"fen"``, ``"move"`` (best move in SAN),
                and optionally ``"moves"`` (dict mapping move -> ``{"score": int}``).
            episode (Episode): Episode produced by ``ChessAgent``.

        Returns:
            EvalOutput: Result with ``legality``, ``accuracy``, and ``move_quality`` signals.
        """
        predicted = _extract_agent_answer(episode).strip()
        board = chess.Board(task["fen"])
        san_map = {board.san(m): m for m in board.legal_moves}
        uci_map = {m.uci(): m for m in board.legal_moves}
        move = san_map.get(predicted) or uci_map.get(predicted)

        if move is None:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal("legality", 0.0), Signal("accuracy", 0.0), Signal("move_quality", 0.0)],
            )

        best_str = task["move"]
        best_move = san_map.get(best_str) or uci_map.get(best_str)
        is_best = move == best_move

        moves_dict: dict | None = task.get("moves")
        if moves_dict and predicted in moves_dict:
            raw = moves_dict[predicted]
            score = int(raw["score"]) if isinstance(raw, dict) else int(raw)
            best_score = max(
                (int(v["score"]) if isinstance(v, dict) else int(v)) for v in moves_dict.values()
            )
            reward = max(0.0, min(1.0, score / best_score)) if best_score > 0 else 0.0
        else:
            reward = 1.0 if is_best else 0.0

        return EvalOutput(
            reward=reward,
            is_correct=is_best,
            signals=[
                Signal("accuracy", 1.0 if is_best else 0.0),
                Signal("legality", 1.0),
                Signal("move_quality", reward),
            ],
        )

from __future__ import annotations

import io
import json
from pathlib import Path

import chess
import chess.pgn
from datasets import load_dataset
from rllm.data import DatasetRegistry


def register_jsonl(path: str | Path, name: str, split: str) -> int:
    """Register a JSONL file as an rllm dataset split.

    Each line must be a JSON object with at least ``"fen"`` and ``"move"`` keys.

    Args:
        path (str | Path): Path to the JSONL file.
        name (str): Dataset name to register under.
        split (str): Split name (e.g. ``"train"``, ``"test"``).

    Returns:
        int: Number of examples registered.
    """
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)


def register_puzzles(path: str | Path, name: str, split: str = "test") -> int:
    """Register a chess_llm-style puzzles JSON file as an rllm dataset split.

    The file format is a list of ``[fen, best_move, legal_moves]`` triples.

    Args:
        path (str | Path): Path to the puzzles JSON file.
        name (str): Dataset name to register under.
        split (str): Split name.

    Returns:
        int: Number of examples registered.
    """
    raw: list[list] = json.loads(Path(path).read_text())
    rows = [{"fen": fen, "move": best_move, "legal_moves": legal} for fen, best_move, legal in raw]
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)


def _positions_from_movetext(
    movetext: str,
    sample_every: int,
    min_ply: int,
    max_ply: int,
) -> list[dict]:
    """Extract sampled positions and played moves from PGN movetext.

    Args:
        movetext (str): PGN movetext string (e.g. ``"1. e4 e5 2. Nf3 ..."``).
        sample_every (int): Yield one position every N plies.
        min_ply (int): First ply to consider (skips openings).
        max_ply (int): Last ply to consider (skips deep endgames).

    Returns:
        list[dict]: Rows with ``"fen"`` and ``"move"`` (SAN) keys.
    """
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    board = game.board()
    rows = []
    for ply, move in enumerate(game.mainline_moves()):
        if min_ply <= ply <= max_ply and ply % sample_every == 0:
            rows.append({"fen": board.fen(), "move": board.san(move)})
        board.push(move)
    return rows


def register_lichess_games(
    name: str,
    split: str,
    max_examples: int = 50_000,
    sample_every: int = 5,
    min_ply: int = 20,
    max_ply: int = 80,
) -> int:
    """Stream positions from ``Lichess/standard-chess-games`` and register as a dataset.

    Samples middlegame positions from human games at regular ply intervals.
    Each row contains the FEN and the move actually played (in SAN notation).

    Args:
        name (str): Dataset name to register under.
        split (str): Split name (e.g. ``"train"``, ``"test"``).
        max_examples (int): Maximum number of positions to collect.
        sample_every (int): Yield one position every N plies per game.
        min_ply (int): First ply to consider (skips openings).
        max_ply (int): Last ply to consider (skips deep endgames).

    Returns:
        int: Number of positions registered.
    """
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    rows: list[dict] = []
    for row in ds:
        rows.extend(_positions_from_movetext(row["movetext"], sample_every, min_ply, max_ply))
        if len(rows) >= max_examples:
            rows = rows[:max_examples]
            break
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)


if __name__ == "__main__":
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    for raw in ds:
        positions = _positions_from_movetext(raw["movetext"], sample_every=5, min_ply=20, max_ply=80)
        if positions:
            print(f"movetext:  {raw['movetext'][:120]}...")
            print(f"extracted: {len(positions)} positions")
            for pos in positions[:3]:
                print(json.dumps(pos))
            break

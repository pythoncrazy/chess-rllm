import json
from pathlib import Path

import chess
from datasets import load_dataset
from rllm.data import DatasetRegistry


def register_jsonl(path: str | Path, name: str, split: str) -> int:
    """Register a JSONL file as an rllm dataset split.

    Each line must be a JSON object with at least ``"fen"`` and ``"move"`` keys.
    Optionally include ``"moves"`` (dict of move -> ``{"score": int}``) for
    proportional reward scoring.

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

    The file format is a list of ``[fen, best_move, legal_moves]`` triples,
    as produced by ``chess_llm/evaluation/puzzles.json``. Moves are in SAN notation.

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


def _lichess_puzzle_to_row(row: dict) -> dict:
    """Convert a Lichess/chess-puzzles row to rllm format.

    The Lichess puzzle format stores the FEN before the opponent's forcing move.
    We apply that first move to reach the actual puzzle position, then return
    the position and the correct response move in SAN.

    Args:
        row (dict): Raw row with ``"FEN"`` (str) and ``"Moves"`` (space-separated UCI str).

    Returns:
        dict: Row with ``"fen"``, ``"move"`` (SAN), ``"rating"``, and ``"themes"``.
    """
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])
    best = chess.Move.from_uci(moves[1])
    return {
        "fen": board.fen(),
        "move": board.san(best),
        "rating": row["Rating"],
        "themes": row["Themes"],
    }


def register_lichess_puzzles(
    name: str,
    split: str,
    min_rating: int = 0,
    max_rating: int = 9999,
    themes: list[str] | None = None,
    max_examples: int | None = None,
) -> int:
    """Load and register puzzles from ``Lichess/chess-puzzles`` on HuggingFace.

    Applies the opponent's first move to get the actual puzzle FEN and stores
    the correct response in SAN notation. Filters by rating range and themes.

    Args:
        name (str): Dataset name to register under.
        split (str): Split name (e.g. ``"train"``, ``"test"``).
        min_rating (int): Minimum puzzle rating (inclusive).
        max_rating (int): Maximum puzzle rating (inclusive).
        themes (list[str] | None): Only keep puzzles that contain at least one
            of these Lichess theme tags (e.g. ``["mateIn1", "fork"]``). ``None``
            keeps all themes.
        max_examples (int | None): Cap the number of registered examples.

    Returns:
        int: Number of examples registered.
    """
    hf_ds = load_dataset("Lichess/chess-puzzles", split="train")
    hf_ds = hf_ds.filter(lambda r: min_rating <= r["Rating"] <= max_rating)
    if themes:
        theme_set = set(themes)
        hf_ds = hf_ds.filter(lambda r: bool(theme_set & set(r["Themes"])))
    if max_examples:
        hf_ds = hf_ds.select(range(min(max_examples, len(hf_ds))))
    rows = [_lichess_puzzle_to_row(r) for r in hf_ds]
    DatasetRegistry.register_dataset(name, rows, split=split)
    return len(rows)

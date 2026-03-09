from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import click
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm


def _analyse(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
) -> dict[str, int]:
    """Score the top N moves for a position using Stockfish.

    Args:
        board (chess.Board): Position to analyse.
        engine (chess.engine.SimpleEngine): Running Stockfish instance.
        depth (int): Search depth per move.
        multipv (int): Number of candidate moves to return.

    Returns:
        dict[str, int]: SAN move to centipawn score from side-to-move POV.
    """
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    return {
        board.san(info["pv"][0]): info["score"].relative.score(mate_score=10000)
        for info in infos
        if info.get("pv") and info["score"].relative.score(mate_score=10000) is not None
    }


def _puzzle_to_row(row: dict) -> dict:
    """Convert a raw Lichess puzzle row to training format.

    Applies the opponent's setup move to reach the puzzle position, then
    returns the FEN and the correct response in SAN.

    Args:
        row (dict): Raw HuggingFace row with ``FEN`` (str), ``Moves``
            (space-separated UCI str), ``Rating`` (int), and ``Themes`` (str).

    Returns:
        dict: Row with ``fen``, ``move``, ``rating``, and ``themes`` keys.
    """
    board = chess.Board(row["FEN"])
    uci_moves = row["Moves"].split()
    board.push_uci(uci_moves[0])
    best = chess.Move.from_uci(uci_moves[1])
    return {
        "fen": board.fen(),
        "move": board.san(best),
        "rating": row["Rating"],
        "themes": row["Themes"],
    }


def _filter_puzzles(
    ds: Dataset,
    min_rating: int,
    max_rating: int,
    themes: str | None,
    max_examples: int | None,
) -> Dataset:
    """Filter a Lichess puzzles dataset by rating, themes, and size.

    Args:
        ds (Dataset): Loaded Lichess/chess-puzzles dataset.
        min_rating (int): Minimum puzzle rating (inclusive).
        max_rating (int): Maximum puzzle rating (inclusive).
        themes (str | None): Comma-separated Lichess theme tags to keep.
        max_examples (int | None): Cap on number of examples.

    Returns:
        Dataset: Filtered dataset.
    """
    ds = ds.filter(lambda r: min_rating <= r["Rating"] <= max_rating)
    if themes:
        theme_set = set(themes.split(","))
        ds = ds.filter(lambda r: bool(theme_set & set(r["Themes"].split())))
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


@click.group()
def cli():
    """Generate chess JSONL training data."""


@cli.command("download")
@click.option("--output", default="data/chess-puzzles", show_default=True, type=click.Path())
def download(output: str) -> None:
    """Download the Lichess chess-puzzles dataset from HuggingFace to disk.

    Args:
        output (str): Directory to save the Arrow dataset to.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Lichess/chess-puzzles", split="train")
    ds.save_to_disk(output)
    click.echo(f"saved {len(ds)} puzzles to {output}")


@cli.command("preprocess")
@click.option("--input", "input_path", default="data/chess-puzzles", show_default=True, type=click.Path(exists=True))
@click.option("--output", default="data/puzzles.jsonl", show_default=True, type=click.Path())
@click.option("--val-output", default=None, type=click.Path())
@click.option("--val-size", default=1000, show_default=True, type=int)
@click.option("--min-rating", default=0, show_default=True, type=int)
@click.option("--max-rating", default=9999, show_default=True, type=int)
@click.option("--themes", default=None, type=str)
@click.option("--max-examples", default=None, type=int)
def preprocess(
    input_path: str,
    output: str,
    val_output: str | None,
    val_size: int,
    min_rating: int,
    max_rating: int,
    themes: str | None,
    max_examples: int | None,
) -> None:
    """Filter and convert downloaded Lichess puzzles to training JSONL.

    Args:
        input_path (str): Path to the Arrow dataset saved by ``download``.
        output (str): Output JSONL file path.
        val_output (str | None): Optional path for val split JSONL.
        val_size (int): Number of examples to hold out for val.
        min_rating (int): Minimum puzzle rating (inclusive).
        max_rating (int): Maximum puzzle rating (inclusive).
        themes (str | None): Comma-separated Lichess theme tags to keep.
        max_examples (int | None): Cap on number of examples to write.
    """
    ds = _filter_puzzles(load_from_disk(input_path), min_rating, max_rating, themes, max_examples)
    rows = [_puzzle_to_row(r) for r in tqdm(ds, desc="preprocessing")]
    if val_output:
        train_rows, val_rows = rows[:-val_size], rows[-val_size:]
        Path(val_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(val_output).open("w") as f:
            for r in val_rows:
                f.write(json.dumps(r) + "\n")
        click.echo(f"wrote {len(val_rows)} val examples to {val_output}")
    else:
        train_rows = rows
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    click.echo(f"wrote {len(train_rows)} train examples to {output}")


@cli.command("puzzles")
@click.option("--input", "input_path", default="data/chess-puzzles", show_default=True, type=click.Path(exists=True))
@click.option("--output", default="data/puzzles_scored.jsonl", show_default=True, type=click.Path())
@click.option("--val-output", default=None, type=click.Path())
@click.option("--val-size", default=1000, show_default=True, type=int)
@click.option("--stockfish", default="stockfish", show_default=True)
@click.option("--depth", default=15, show_default=True, type=int)
@click.option("--multipv", default=5, show_default=True, type=int)
@click.option("--min-rating", default=1000, show_default=True, type=int)
@click.option("--max-rating", default=2200, show_default=True, type=int)
@click.option("--themes", default=None, type=str)
@click.option("--max-examples", default=None, type=int)
def from_puzzles(
    input_path: str,
    output: str,
    val_output: str | None,
    val_size: int,
    stockfish: str,
    depth: int,
    multipv: int,
    min_rating: int,
    max_rating: int,
    themes: str | None,
    max_examples: int | None,
) -> None:
    """Enrich downloaded Lichess puzzles with Stockfish scores and save as JSONL.

    Args:
        input_path (str): Path to the Arrow dataset saved by ``download``.
        output (str): Output JSONL file path.
        val_output (str | None): Optional path for val split JSONL.
        val_size (int): Number of examples to hold out for val.
        stockfish (str): Path to the Stockfish binary.
        depth (int): Stockfish search depth.
        multipv (int): Number of candidate moves to score per position.
        min_rating (int): Minimum puzzle rating (inclusive).
        max_rating (int): Maximum puzzle rating (inclusive).
        themes (str | None): Comma-separated Lichess theme tags to keep.
        max_examples (int | None): Cap on number of examples to write.
    """
    ds = _filter_puzzles(load_from_disk(input_path), min_rating, max_rating, themes, max_examples)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    rows = []
    for row in tqdm(ds, desc="analysing"):
        board = chess.Board(row["FEN"])
        uci_moves = row["Moves"].split()
        board.push_uci(uci_moves[0])
        best = chess.Move.from_uci(uci_moves[1])
        move_scores = _analyse(board, engine, depth, multipv)
        if not move_scores:
            continue
        rows.append({
            "fen": board.fen(),
            "move": board.san(best),
            "moves": {san: {"score": score} for san, score in move_scores.items()},
            "rating": row["Rating"],
            "themes": row["Themes"],
        })
    engine.quit()
    if val_output:
        train_rows, val_rows = rows[:-val_size], rows[-val_size:]
        Path(val_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(val_output).open("w") as f:
            for r in val_rows:
                f.write(json.dumps(r) + "\n")
        click.echo(f"wrote {len(val_rows)} val examples to {val_output}")
    else:
        train_rows = rows
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    click.echo(f"wrote {len(train_rows)} train examples to {output}")


@cli.command("pgn")
@click.argument("pgn_path", type=click.Path(exists=True))
@click.option("--output", default="data/pgn.jsonl", show_default=True, type=click.Path())
@click.option("--stockfish", default="stockfish", show_default=True)
@click.option("--depth", default=15, show_default=True, type=int)
@click.option("--multipv", default=5, show_default=True, type=int)
@click.option("--sample-every", default=5, show_default=True, type=int)
@click.option("--min-ply", default=20, show_default=True, type=int)
@click.option("--max-ply", default=80, show_default=True, type=int)
@click.option("--min-gap", default=50, show_default=True, type=int)
@click.option("--max-games", default=None, type=int)
def from_pgn(
    pgn_path: str,
    output: str,
    stockfish: str,
    depth: int,
    multipv: int,
    sample_every: int,
    min_ply: int,
    max_ply: int,
    min_gap: int,
    max_games: int | None,
) -> None:
    """Extract and score middlegame positions from a PGN file.

    Samples one position every ``sample_every`` plies, skipping openings and
    deep endgames. Keeps only positions where the score gap between the best
    and second-best move is at least ``min_gap`` centipawns.

    Args:
        pgn_path (str): Path to the input PGN file.
        output (str): Output JSONL file path.
        stockfish (str): Path to the Stockfish binary.
        depth (int): Stockfish search depth.
        multipv (int): Number of candidate moves to score per position.
        sample_every (int): Sample one position every N plies.
        min_ply (int): Skip positions below this ply count.
        max_ply (int): Skip positions above this ply count.
        min_gap (int): Minimum centipawn gap between 1st and 2nd best move.
        max_games (int | None): Stop after this many games.
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    n_written = 0
    n_games = 0
    with Path(output).open("w") as f, open(pgn_path) as pgn_file:
        for _ in tqdm(iter(int, 1), desc="games"):
            game = chess.pgn.read_game(pgn_file)
            if game is None or (max_games and n_games >= max_games):
                break
            n_games += 1
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply < min_ply or ply > max_ply or ply % sample_every != 0:
                    continue
                move_scores = _analyse(board, engine, depth, multipv)
                if len(move_scores) < 2:
                    continue
                top_scores = sorted(move_scores.values(), reverse=True)
                if top_scores[0] - top_scores[1] < min_gap:
                    continue
                best_san = max(move_scores, key=move_scores.__getitem__)
                f.write(json.dumps({
                    "fen": board.fen(),
                    "move": best_san,
                    "moves": {san: {"score": score} for san, score in move_scores.items()},
                }) + "\n")
                n_written += 1
    engine.quit()
    click.echo(f"wrote {n_written} positions from {n_games} games to {output}")


if __name__ == "__main__":
    cli()

"""Generate SFT data from Stockfish vs Stockfish games seeded from an opening book.

Each game starts from a random 3-move opening, with a small fraction of moves
randomized to diversify positions. Positions are sampled every N plies and
analyzed with Stockfish multipv=3.

Usage:
    uv run python generate_sf_vs_sf_data.py --n 200000 --out data/sf_train_stockfish_vs_stockfish.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_USER_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_SUFFIX = "\n\nGive your answer in <answer>...</answer> tags."
_MAX_CP = 10_000


def _get_openings() -> list[str]:
    """Download and return FEN strings from the Stockfish 3-move opening book.

    Returns:
        list[str]: List of opening position FEN strings.
    """
    url = "https://github.com/official-stockfish/books/raw/refs/heads/master/noob_3moves.epd.zip"
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("noob_3moves.epd") as f:
            return [ln.strip() for ln in f.read().decode().splitlines() if ln.strip()]


def _play_game(
    opening_fen: str,
    engine: chess.engine.SimpleEngine,
    move_time: float,
    variation_rate: float,
) -> list[str]:
    """Play a Stockfish vs Stockfish game from an opening FEN and collect positions.

    Args:
        opening_fen (str): Starting FEN position (from opening book).
        engine (chess.engine.SimpleEngine): Stockfish engine instance.
        move_time (float): Time limit in seconds per move for game play.
        variation_rate (float): Fraction of moves to replace with a random legal move.

    Returns:
        list[str]: FEN strings of all positions encountered during the game.
    """
    board = chess.Board(opening_fen)
    fens: list[str] = []
    while not board.is_game_over(claim_draw=True):
        fens.append(board.fen())
        if random.random() < variation_rate:
            move = random.choice(list(board.legal_moves))
        else:
            result = engine.play(board, chess.engine.Limit(time=move_time))
            move = result.move or random.choice(list(board.legal_moves))
        board.push(move)
    return fens


def _top3_cot(top3: list[str]) -> str:
    """Return shuffled top-3 moves as a CoT prefix with 5% noise, matching SFT format.

    Args:
        top3 (list[str]): Top-3 moves in SAN notation sorted best-first.

    Returns:
        str: Comma-separated shuffled moves followed by ``</think>``.
    """
    shuffled = top3.copy()
    random.shuffle(shuffled)
    for i in range(len(shuffled)):
        if random.random() < 0.05:
            shuffled[i] = random.choice(top3)
    return ", ".join(shuffled) + "</think>"


def _analyze_positions(
    fens: list[str],
    engine: chess.engine.SimpleEngine,
    analysis_time: float,
) -> list[dict]:
    """Analyze a list of FEN positions with Stockfish multipv=3.

    Args:
        fens (list[str]): FEN strings to analyze.
        engine (chess.engine.SimpleEngine): Stockfish engine instance.
        analysis_time (float): Time limit in seconds per position for analysis.

    Returns:
        list[dict]: Rows with ``fen``, ``move``, ``moves``, and ``messages`` keys.
    """
    rows: list[dict] = []
    for fen in fens:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        infos = engine.analyse(board, chess.engine.Limit(time=analysis_time), multipv=3)
        moves: dict[str, dict] = {}
        for info in infos:
            if "pv" not in info or not info["pv"]:
                continue
            san = board.san(info["pv"][0])
            pov = info["score"].relative
            moves[san] = {"score": _MAX_CP if pov.is_mate() else (pov.score() or 0)}
        if not moves:
            continue
        best = max(moves, key=lambda s: moves[s]["score"])
        top3 = sorted(moves, key=lambda s: moves[s]["score"], reverse=True)
        rows.append({
            "fen": fen,
            "move": best,
            "moves": moves,
            "messages": [
                {"role": "user", "content": _USER_PREFIX + fen + _USER_SUFFIX},
                {"role": "assistant", "content": _top3_cot(top3) + f"<answer>{best}</answer>"},
            ],
        })
    return rows


_worker_engine: chess.engine.SimpleEngine | None = None
_worker_move_time: float = 0.25
_worker_analysis_time: float = 0.25
_worker_variation_rate: float = 0.05
_worker_sample_every: int = 3


def _worker_init(move_time: float, analysis_time: float, variation_rate: float, sample_every: int) -> None:
    """Pool initializer: start a persistent Stockfish engine for this worker.

    Args:
        move_time (float): Seconds per move during game play.
        analysis_time (float): Seconds per position for multipv=3 analysis.
        variation_rate (float): Fraction of game moves to randomize.
        sample_every (int): Sample 1 position every N plies per game.
    """
    global _worker_engine, _worker_move_time, _worker_analysis_time, _worker_variation_rate, _worker_sample_every
    random.seed(os.getpid())
    _worker_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _worker_move_time = move_time
    _worker_analysis_time = analysis_time
    _worker_variation_rate = variation_rate
    _worker_sample_every = sample_every


def _worker(opening: str) -> list[dict]:
    """Worker: play one game from an opening and analyze sampled positions.

    Args:
        opening (str): Starting FEN position.

    Returns:
        list[dict]: Analyzed position rows from this game.
    """
    assert _worker_engine is not None
    all_fens = _play_game(opening, _worker_engine, _worker_move_time, _worker_variation_rate)
    return _analyze_positions(all_fens[::_worker_sample_every], _worker_engine, _worker_analysis_time)


def main() -> None:
    """Entry point: generate Stockfish vs Stockfish SFT data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000, help="Target number of examples")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--game-time", type=float, default=0.25, help="Seconds per move during game play")
    parser.add_argument("--analysis-time", type=float, default=0.25, help="Seconds per position for multipv=3 analysis")
    parser.add_argument("--variation-rate", type=float, default=0.05, help="Fraction of game moves to randomize")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample 1 position every N plies per game")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading opening book...")
    openings = _get_openings()
    print(f"Loaded {len(openings)} openings.")

    positions_per_game = max(1, 80 // args.sample_every)
    n_games = max(args.workers, -(-args.n * 2 // positions_per_game))
    game_list = [random.choice(openings) for _ in range(n_games)]

    initargs = (args.game_time, args.analysis_time, args.variation_rate, args.sample_every)
    written = 0
    with out_path.open("w") as f, tqdm(total=args.n, unit="pos") as bar:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=initargs) as pool:
            for rows in pool.imap_unordered(_worker, game_list):
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    written += 1
                    bar.update(1)
                    if written >= args.n:
                        pool.terminate()
                        break
                if written >= args.n:
                    break

    print(f"Done. Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()

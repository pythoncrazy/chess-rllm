"""Generate SFT data from Stockfish vs Stockfish games seeded from an opening book.

Each game starts from either a random 5-move opening (from the official Stockfish
book) or a random-walk position, with a small fraction of moves randomized to
diversify positions. Positions are sampled every N plies and analyzed with
Stockfish multipv=3. All collected rows are shuffled before writing so that
training batches are positionally varied.

Usage:
    uv run python datagen.py --n 500000 --out data/sf_train.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.engine

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_USER_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_SUFFIX = "\n\nGive your answer in UCI notation (e.g. g1f3) in <answer>...</answer> tags."
_MAX_CP = 10_000


_OPENING_BOOK_CACHE = Path(__file__).parent / "data" / "noob_5moves.epd"
_OPENING_BOOK_URL = "https://github.com/official-stockfish/books/raw/refs/heads/master/noob_5moves.epd.zip"


def _get_openings() -> list[str]:
    """Return FEN strings from the Stockfish 5-move opening book.

    Downloads once and caches to data/noob_5moves.epd; subsequent calls
    load from disk instantly.

    Returns:
        list[str]: List of opening position FEN strings.
    """
    if _OPENING_BOOK_CACHE.exists():
        lines = _OPENING_BOOK_CACHE.read_text().splitlines()
    else:
        print(f"Downloading opening book from {_OPENING_BOOK_URL} ...")
        _OPENING_BOOK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(_OPENING_BOOK_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with zf.open("noob_5moves.epd") as f:
                content = f.read().decode()
        _OPENING_BOOK_CACHE.write_text(content)
        lines = content.splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _random_walk_opening(n_moves: int = 8) -> str:
    """Generate a starting FEN by playing n_moves random legal moves from the start.

    Args:
        n_moves (int): Number of random moves to play (default 8, ~4 moves per side).

    Returns:
        str: FEN of the resulting position, or the starting FEN if game ends early.
    """
    board = chess.Board()
    for _ in range(n_moves):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board.fen()


def _play_game(
    opening_fen: str,
    engine: chess.engine.SimpleEngine,
    move_time: float,
    variation_rate: float,
) -> tuple[list[str], chess.Board]:
    """Play a Stockfish vs Stockfish game from an opening FEN and collect positions.

    Args:
        opening_fen (str): Starting FEN position (from opening book).
        engine (chess.engine.SimpleEngine): Stockfish engine instance.
        move_time (float): Time limit in seconds per move for game play.
        variation_rate (float): Fraction of moves to replace with a random legal move.

    Returns:
        tuple[list[str], chess.Board]: FEN strings of all positions and the final board.
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
    return fens, board


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
            mv = info["pv"][0]
            uci = mv.uci()
            pov = info["score"].relative
            moves[uci] = {"score": _MAX_CP if pov.is_mate() else (pov.score() or 0)}
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
    _worker_engine.configure({"Skill Level": random.randint(0, 20)})
    all_fens, _ = _play_game(opening, _worker_engine, _worker_move_time, _worker_variation_rate)
    return _analyze_positions(all_fens[::_worker_sample_every], _worker_engine, _worker_analysis_time)


def main() -> None:
    """Entry point: generate Stockfish vs Stockfish SFT data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500_000, help="Target number of examples")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--game-time", type=float, default=0.25, help="Seconds per move during game play")
    parser.add_argument("--analysis-time", type=float, default=0.25, help="Seconds per position for multipv=3 analysis")
    parser.add_argument("--variation-rate", type=float, default=0.05, help="Fraction of game moves to randomize")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample 1 position every N plies per game")
    parser.add_argument("--random-walk-ratio", type=float, default=0.5,
                        help="Fraction of games seeded from random walks vs the opening book (default 0.5)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading opening book...")
    openings = _get_openings()
    print(f"Loaded {len(openings)} openings.")

    positions_per_game = max(1, 80 // args.sample_every)
    n_games = max(args.workers, -(-args.n * 2 // positions_per_game))
    n_random = int(n_games * args.random_walk_ratio)
    n_book = n_games - n_random
    game_list = (
        [random.choice(openings) for _ in range(n_book)]
        + [_random_walk_opening() for _ in range(n_random)]
    )
    random.shuffle(game_list)
    print(f"Game seeds: {n_book} from 5-move book, {n_random} from random walks ({n_games} total).")

    initargs = (args.game_time, args.analysis_time, args.variation_rate, args.sample_every)
    written = 0
    t_start = time.time()

    with out_path.open("w") as f:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=initargs) as pool:
            for rows in pool.imap_unordered(_worker, game_list):
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    written += 1
                    if written % 10_000 == 0:
                        elapsed = time.time() - t_start
                        rate = written / elapsed
                        eta = (args.n - written) / rate if rate > 0 else 0
                        print(f"[{written:>10,} / {args.n:,}] "
                              f"{elapsed:6.0f}s elapsed  "
                              f"{rate:7.0f} pos/s  "
                              f"ETA {eta/3600:.1f}h",
                              flush=True)
                    if written >= args.n:
                        pool.terminate()
                        break
                if written >= args.n:
                    break

    # Shuffle in-place: read all lines, shuffle, rewrite.
    print("Shuffling output...")
    lines = out_path.read_text().splitlines()
    random.shuffle(lines)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Done. Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()

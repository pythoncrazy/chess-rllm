"""Generate GRPO training data by playing full games from the UHO_4060_v4 opening book.

Usage:
    uv run python scripts/gen_opening_grpo.py --out data/grpo_openings.jsonl --n 200000
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
_CACHE = Path(__file__).parent.parent / "data" / "UHO_4060_v4.epd"
_URL = "https://raw.githubusercontent.com/official-stockfish/books/master/UHO_4060_v4.epd.zip"
_MAX_CP = 10_000

_engine: chess.engine.SimpleEngine | None = None
_move_time: float = 0.05
_analysis_time: float = 0.25
_variation_rate: float = 0.05
_sample_every: int = 3


def _load_epd() -> list[str]:
    """Load UHO EPD positions, downloading and caching if needed.

    Returns:
        list[str]: FEN strings (one per opening line).
    """
    if not _CACHE.exists():
        print(f"Downloading {_URL} ...")
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            content = zf.read(zf.namelist()[0]).decode()
        _CACHE.write_text(content)
    else:
        content = _CACHE.read_text()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return [" ".join(ln.split()[:4]) + " 0 1" for ln in lines]


def _worker_init(move_time: float, analysis_time: float, variation_rate: float, sample_every: int) -> None:
    """Initialize a persistent Stockfish engine for this worker process.

    Args:
        move_time (float): Seconds per move during game play.
        analysis_time (float): Seconds per position for multipv=3 analysis.
        variation_rate (float): Fraction of game moves to randomize.
        sample_every (int): Sample 1 position every N plies per game.
    """
    global _engine, _move_time, _analysis_time, _variation_rate, _sample_every
    random.seed(os.getpid())
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _move_time = move_time
    _analysis_time = analysis_time
    _variation_rate = variation_rate
    _sample_every = sample_every


def _worker(fen: str) -> list[dict]:
    """Play one full game from a UHO opening FEN and analyze sampled positions.

    Args:
        fen (str): Starting FEN position from UHO opening book.

    Returns:
        list[dict]: Analyzed position rows with fen, move, moves, messages keys.
    """
    assert _engine is not None
    try:
        board = chess.Board(fen)
        fens: list[str] = []
        while not board.is_game_over(claim_draw=True):
            fens.append(board.fen())
            if random.random() < _variation_rate:
                move = random.choice(list(board.legal_moves))
            else:
                result = _engine.play(board, chess.engine.Limit(time=_move_time))
                move = result.move or random.choice(list(board.legal_moves))
            board.push(move)

        rows: list[dict] = []
        for pos_fen in fens[::_sample_every]:
            pos_board = chess.Board(pos_fen)
            if pos_board.is_game_over():
                continue
            infos = _engine.analyse(pos_board, chess.engine.Limit(time=_analysis_time), multipv=3)
            moves: dict[str, dict] = {}
            for info in infos:
                if "pv" not in info or not info["pv"]:
                    continue
                mv = info["pv"][0]
                pov = info["score"].relative
                moves[pos_board.san(mv)] = {"score": _MAX_CP if pov.is_mate() else (pov.score() or 0)}
            if not moves:
                continue
            top3 = sorted(moves, key=lambda s: moves[s]["score"], reverse=True)
            rows.append({"fen": pos_fen, "move": top3[0], "moves": moves})
        return rows
    except chess.engine.EngineError:
        return []


def main() -> None:
    """Generate GRPO opening dataset from UHO_4060_v4 full games."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--move-time", type=float, default=0.05, help="Seconds per move during game play")
    parser.add_argument("--analysis-time", type=float, default=0.25)
    parser.add_argument("--variation-rate", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=3)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fens = _load_epd()
    print(f"Loaded {len(fens)} UHO positions.")
    random.shuffle(fens)

    positions_per_game = max(1, 80 // args.sample_every)
    n_games = max(args.workers, -(-args.n * 2 // positions_per_game))
    game_fens = [fens[i % len(fens)] for i in range(n_games)]
    random.shuffle(game_fens)
    print(f"Playing {n_games} games (est. {positions_per_game} pos/game).")

    initargs = (args.move_time, args.analysis_time, args.variation_rate, args.sample_every)
    written = 0
    t_start = time.time()

    with out_path.open("w") as f, mp.Pool(args.workers, _worker_init, initargs) as pool:
        for rows in pool.imap_unordered(_worker, game_fens):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                written += 1
                if written % 10_000 == 0:
                    elapsed = time.time() - t_start
                    rate = written / elapsed
                    print(f"[{written:>7,} / {args.n:,}]  {rate:.0f} pos/s  "
                          f"ETA {(args.n - written) / rate / 60:.1f}m", flush=True)
            if written >= args.n:
                pool.terminate()
                break

    lines = out_path.read_text().splitlines()
    random.shuffle(lines)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Done. {len(lines)} rows → {out_path}")


if __name__ == "__main__":
    main()

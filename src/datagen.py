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
_DATA = Path(__file__).parent.parent / "data"
_MAX_CP = 10_000

_BOOKS: list[tuple[str, str]] = [
    ("noob_5moves.epd", "https://github.com/official-stockfish/books/raw/refs/heads/master/noob_5moves.epd.zip"),
    ("UHO_4060_v4.epd", "https://raw.githubusercontent.com/official-stockfish/books/master/UHO_4060_v4.epd.zip"),
]

_engine: chess.engine.SimpleEngine | None = None
_move_time: float = 0.05
_analysis_time: float = 0.1
_variation_rate: float = 0.05
_sample_every: int = 3


def _load_book(filename: str, url: str) -> list[str]:
    path = _DATA / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(url).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            content = zf.read(zf.namelist()[0]).decode()
        path.write_text(content)
    else:
        content = path.read_text()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return [" ".join(ln.split()[:4]) + " 0 1" for ln in lines]


def _random_walk(n: int = 8) -> str:
    board = chess.Board()
    for _ in range(n):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board.fen()


def _worker_init(move_time: float, analysis_time: float, variation_rate: float, sample_every: int) -> None:
    global _engine, _move_time, _analysis_time, _variation_rate, _sample_every
    random.seed(os.getpid())
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _move_time, _analysis_time, _variation_rate, _sample_every = move_time, analysis_time, variation_rate, sample_every


def _worker(fen: str) -> list[dict]:
    assert _engine is not None
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
    for pos in fens[::_sample_every]:
        b = chess.Board(pos)
        if b.is_game_over():
            continue
        infos = _engine.analyse(b, chess.engine.Limit(time=_analysis_time), multipv=3)
        moves: dict[str, dict] = {}
        for info in infos:
            if "pv" not in info or not info["pv"]:
                continue
            pov = info["score"].relative
            moves[b.san(info["pv"][0])] = {"score": _MAX_CP if pov.is_mate() else (pov.score() or 0)}
        if not moves:
            continue
        top3 = sorted(moves, key=lambda s: moves[s]["score"], reverse=True)
        best = top3[0]
        rows.append({"fen": pos, "move": best, "moves": moves})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--move-time", type=float, default=0.05)
    parser.add_argument("--analysis-time", type=float, default=0.1)
    parser.add_argument("--variation-rate", type=float, default=0.05)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--random-walk-ratio", type=float, default=0.34)
    args = parser.parse_args()

    all_openings: list[str] = []
    for filename, url in _BOOKS:
        book = _load_book(filename, url)
        print(f"Loaded {len(book):,} positions from {filename}")
        all_openings.extend(book)

    positions_per_game = max(1, 80 // args.sample_every)
    n_games = max(args.workers, -(-args.n * 2 // positions_per_game))
    n_random = int(n_games * args.random_walk_ratio)
    n_book = n_games - n_random
    game_list = [random.choice(all_openings) for _ in range(n_book)] + [_random_walk() for _ in range(n_random)]
    random.shuffle(game_list)
    print(f"{n_book:,} book + {n_random:,} random-walk seeds ({n_games:,} games total)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    initargs = (args.move_time, args.analysis_time, args.variation_rate, args.sample_every)
    written = 0
    t_start = time.time()

    with out_path.open("w") as f, mp.Pool(args.workers, _worker_init, initargs) as pool:
        for rows in pool.imap_unordered(_worker, game_list):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                written += 1
                if written % 10_000 == 0:
                    elapsed = time.time() - t_start
                    rate = written / elapsed
                    print(f"[{written:>10,} / {args.n:,}]  {rate:.0f} pos/s  ETA {(args.n-written)/rate/3600:.1f}h", flush=True)
            if written >= args.n:
                pool.terminate()
                break

    print("Shuffling...")
    lines = out_path.read_text().splitlines()
    random.shuffle(lines)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Done. {written:,} rows → {out_path}")


if __name__ == "__main__":
    main()

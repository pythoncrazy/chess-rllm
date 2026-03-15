"""Generate SFT training data: FEN positions + Stockfish best moves.

Uses multiprocessing to saturate all CPUs on the TPU VM.

Usage:
    uv run python generate_sft_data.py --n 200000 --out data/sft_train.jsonl
    uv run python generate_sft_data.py --n 5000 --out data/sft_val.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")

# Prompt format matching eval.py
_USER_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_SUFFIX = "\n\nGive your answer in <answer>...</answer> tags."

_MAX_CENTIPAWNS = 10_000  # cap for mate scores


def _top3_cot(top3_sans: list[str]) -> str:
    """Return shuffled top-3 candidate moves as a CoT prefix (chess_llm style).

    Applies 5 % noise: each candidate has a 5 % chance of being replaced by a
    random other candidate, encouraging the model to reason rather than memorise.
    """
    shuffled = top3_sans.copy()
    random.shuffle(shuffled)
    for i in range(len(shuffled)):
        if random.random() < 0.05:
            shuffled[i] = random.choice(top3_sans)
    return ", ".join(shuffled) + "</think>"


def _positions_from_pgn(movetext: str, sample_every: int, min_ply: int, max_ply: int) -> list[str]:
    """Return FEN strings sampled from a PGN game."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []
    board = game.board()
    fens = []
    for ply, move in enumerate(game.mainline_moves()):
        if min_ply <= ply <= max_ply and ply % sample_every == 0:
            fens.append(board.fen())
        board.push(move)
    return fens


def _worker_init():
    """Called once per worker process — nothing to do but could seed RNG."""
    random.seed(os.getpid())


def _analyze_batch(fens: list[str]) -> list[dict]:
    """Analyze a batch of FENs, producing top-3 scored moves + CoT messages."""
    results = []
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            for fen in fens:
                try:
                    board = chess.Board(fen)
                    if board.is_game_over():
                        continue
                    infos = engine.analyse(board, chess.engine.Limit(time=0.25), multipv=3)
                    moves: dict[str, dict] = {}
                    for info in infos:
                        if "pv" not in info or not info["pv"]:
                            continue
                        mv = info["pv"][0]
                        san = board.san(mv)
                        pov = info["score"].relative
                        cp = _MAX_CENTIPAWNS if pov.is_mate() else pov.score()
                        moves[san] = {"score": cp}
                    if not moves:
                        continue
                    best_san = max(moves, key=lambda s: moves[s]["score"])
                    top3 = sorted(moves, key=lambda s: moves[s]["score"], reverse=True)
                    user_content = _USER_PREFIX + fen + _USER_SUFFIX
                    assistant_content = _top3_cot(top3) + f"<answer>{best_san}</answer>"
                    results.append({
                        "fen": fen,
                        "move": best_san,
                        "moves": moves,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                    })
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Worker engine error: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Stockfish SFT data")
    parser.add_argument("--n", type=int, default=50_000, help="Target number of examples")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=min(mp.cpu_count(), 200), help="Worker processes")
    parser.add_argument("--batch-size", type=int, default=20, help="FENs per worker batch")
    parser.add_argument("--sample-every", type=int, default=3, help="Sample every N plies")
    parser.add_argument("--min-ply", type=int, default=10, help="Min ply to sample")
    parser.add_argument("--max-ply", type=int, default=100, help="Max ply to sample")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Lichess games (target {args.n} examples, {args.workers} workers)...")

    # Collect FENs from Lichess games
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    all_fens: list[str] = []
    needed_fens = args.n * 2  # over-collect to account for filtering

    for row in ds:
        batch = _positions_from_pgn(row["movetext"], args.sample_every, args.min_ply, args.max_ply)
        all_fens.extend(batch)
        if len(all_fens) >= needed_fens:
            break
        if len(all_fens) % 10_000 == 0 and len(all_fens) > 0:
            logger.info(f"  Collected {len(all_fens)} FENs so far...")

    random.shuffle(all_fens)
    all_fens = all_fens[:needed_fens]
    logger.info(f"Collected {len(all_fens)} FENs, analyzing with {args.workers} workers...")

    # Split into batches for workers
    batches = [all_fens[i:i + args.batch_size] for i in range(0, len(all_fens), args.batch_size)]

    written = 0
    analyzed = 0
    total_batches = len(batches)
    with out_path.open("w") as f:
        with mp.Pool(args.workers, initializer=_worker_init) as pool:
            for i, examples in enumerate(pool.imap_unordered(_analyze_batch, batches, chunksize=1)):
                analyzed += args.batch_size
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
                    written += 1
                    if written >= args.n:
                        pool.terminate()
                        break
                if written >= args.n:
                    break
                pct = 100 * (i + 1) / total_batches
                kept_rate = written / analyzed if analyzed else 0
                sys.stdout.write(
                    f"\r  [{i+1}/{total_batches} batches | {pct:.1f}%]  "
                    f"written {written}/{args.n}  "
                    f"keep rate {kept_rate:.1%}  "
                    f"workers {args.workers}"
                )
                sys.stdout.flush()
    sys.stdout.write("\n")

    logger.info(f"Done. Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()

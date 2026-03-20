"""Generate English chess explanations via Gemini 3.1 Flash-Lite.

Cost: ~$49 for 80k samples (20M input @ $0.25/1M + 30M output @ $1.50/1M).

Usage:
    uv run python src/gen_english_data.py --out data/english_80k.jsonl --n 80000
"""
from __future__ import annotations

import argparse
import asyncio
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
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt import format_prompt

load_dotenv()

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_DATA = Path(__file__).parent.parent / "data"
_MAX_CP = 10_000
_BOOKS: list[tuple[str, str]] = [
    ("noob_5moves.epd", "https://github.com/official-stockfish/books/raw/refs/heads/master/noob_5moves.epd.zip"),
    ("UHO_4060_v4.epd", "https://raw.githubusercontent.com/official-stockfish/books/master/UHO_4060_v4.epd.zip"),
]
_engine: chess.engine.SimpleEngine | None = None
_analysis_time: float = 0.2


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


def _worker_init(analysis_time: float) -> None:
    global _engine, _analysis_time
    random.seed(os.getpid())
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    _analysis_time = analysis_time


def _worker(fen: str) -> list[dict]:
    assert _engine is not None
    board = chess.Board(fen)
    fens: list[str] = []
    while not board.is_game_over(claim_draw=True):
        fens.append(board.fen())
        result = _engine.play(board, chess.engine.Limit(time=0.05))
        board.push(result.move or random.choice(list(board.legal_moves)))

    rows: list[dict] = []
    for pos in fens[::5]:
        b = chess.Board(pos)
        if b.is_game_over():
            continue
        infos = _engine.analyse(b, chess.engine.Limit(time=_analysis_time), multipv=5)
        moves: dict[str, dict] = {}
        pvs: dict[str, list[str]] = {}
        for info in infos:
            if "pv" not in info or not info["pv"]:
                continue
            mv = info["pv"][0]
            san = b.san(mv)
            pov = info["score"].relative
            moves[san] = {"score": _MAX_CP if pov.is_mate() else (pov.score() or 0)}
            pv_board = b.copy()
            pv_sans: list[str] = []
            for pv_mv in info["pv"][:4]:
                if pv_board.is_game_over():
                    break
                pv_sans.append(pv_board.san(pv_mv))
                pv_board.push(pv_mv)
            pvs[san] = pv_sans
        if not moves:
            continue
        top = sorted(moves, key=lambda s: moves[s]["score"], reverse=True)
        rows.append({"fen": pos, "best": top[0], "moves": moves, "pvs": pvs})
    return rows


def _build_prompt(row: dict) -> str:
    board = chess.Board(row["fen"])
    side = "White" if board.turn == chess.WHITE else "Black"
    top = sorted(row["moves"], key=lambda s: row["moves"][s]["score"], reverse=True)
    lines = [
        f"Position (FEN): {row['fen']}",
        f"Side to move: {side}",
        "Stockfish analysis (best moves):",
    ]
    for i, san in enumerate(top[:5], 1):
        score = row["moves"][san]["score"]
        cp_str = f"+{score/100:.2f}" if score >= 0 else f"{score/100:.2f}"
        pv = " ".join(row["pvs"].get(san, []))
        lines.append(f"  {i}. {san}  {cp_str}  PV: {pv}")
    lines.append(
        f"\nAs a grandmaster, reason step by step through the top candidate moves "
        f"and explain why {top[0]} is best. End EXACTLY with: <answer>{top[0]}</answer>"
    )
    return "\n".join(lines)


async def _call_gemini(
    client: genai.Client,
    sem: asyncio.Semaphore,
    row: dict,
) -> dict | None:
    prompt = _build_prompt(row)
    user_content = format_prompt(row["fen"])
    async with sem:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.8,
            ),
        )
    text = (response.text or "").strip()
    if f"<answer>{row['best']}</answer>" not in text and f"<move>{row['best']}</move>" not in text:
        return None
    if text.startswith("<think>"):
        text = text[len("<think>"):]
    tag = f"<move>{row['best']}</move>" if f"<move>{row['best']}</move>" in text else f"<answer>{row['best']}</answer>"
    replacement = f"<move>{row['best']}</move>"
    if "</think>" not in text:
        text = text.replace(tag, f"</think>{replacement}", 1)
    else:
        text = text.replace(tag, replacement, 1)
    return {
        "fen": row["fen"],
        "move": row["best"],
        "moves": row["moves"],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": text},
        ],
    }


async def _run_async(rows: list[dict], out_path: Path, concurrency: int) -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    written = 0
    t_start = time.time()
    with out_path.open("w") as f:
        tasks = [_call_gemini(client, sem, row) for row in rows]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                continue
            f.write(json.dumps(result) + "\n")
            f.flush()
            written += 1
            if written % 1000 == 0:
                elapsed = time.time() - t_start
                rate = written / elapsed
                remaining = (len(rows) - written) / rate / 60
                print(f"[{written:>6,} / {len(rows):,}]  {rate:.1f} req/s  ETA {remaining:.1f}m", flush=True)
    print(f"Done. {written:,} rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=80_000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--analysis-time", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=30, help="Gemini API concurrency")
    args = parser.parse_args()

    all_openings: list[str] = []
    for filename, url in _BOOKS:
        book = _load_book(filename, url)
        print(f"Loaded {len(book):,} positions from {filename}")
        all_openings.extend(book)

    positions_per_game = max(1, 80 // 5)
    n_games = max(args.workers, -(-args.n * 2 // positions_per_game))
    game_fens = [random.choice(all_openings) for _ in range(n_games)]
    random.shuffle(game_fens)
    print(f"Playing {n_games:,} games to collect ~{args.n:,} positions...")

    all_rows: list[dict] = []
    initargs = (args.analysis_time,)
    with mp.Pool(args.workers, _worker_init, initargs) as pool:
        for rows in pool.imap_unordered(_worker, game_fens):
            all_rows.extend(rows)
            if len(all_rows) >= args.n * 2:
                pool.terminate()
                break

    random.shuffle(all_rows)
    rows = all_rows[: args.n]
    print(f"Collected {len(rows):,} positions. Starting Gemini API calls...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_async(rows, out_path, args.concurrency))


if __name__ == "__main__":
    main()

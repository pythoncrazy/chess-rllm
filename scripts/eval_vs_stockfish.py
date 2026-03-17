"""Evaluate a GRPO/SFT checkpoint by playing full games against Stockfish.

Plays N games per Stockfish depth (1-20), alternating colours, with all depths
running in parallel (one thread per depth, each with its own engine instance).
Reports win / draw / loss rates for each depth. Each game is logged in full
to a JSONL file (one game per line) for post-hoc inspection.

Usage:
    uv run python scripts/eval_vs_stockfish.py \\
        --checkpoint tinker://...sampler_weights/final \\
        --games-per-depth 20 \\
        --depths 1 2 3 4 5 6 7 8 9 10

The checkpoint must point to sampler_weights (not weights), e.g.:
    tinker://<id>:train:0/sampler_weights/final
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import chess
import chess.engine
import tinker
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
MODEL_NAME = "moonshotai/Kimi-K2.5"

_USER_PREFIX = (
    "You are a grandmaster chess player. "
    "What should I respond in the following position, given in FEN notation?  "
)
_USER_SUFFIX = "\n\nGive your answer in UCI notation (e.g. g1f3) in <answer>...</answer> tags."

_ANSWER_RE = re.compile(r"<answer>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</answer>", re.IGNORECASE)


@dataclass
class MoveRecord:
    ply: int                  # half-move number (0-indexed)
    side: str                 # "model" or "stockfish"
    fen_before: str           # FEN before the move
    raw_response: str         # model's full decoded output (empty for SF)
    uci_parsed: str | None    # UCI move parsed from response (None if SF or failed parse)
    uci_played: str           # actual UCI move played (may be fallback)
    legal: bool               # whether the parsed move was legal (always True for SF)


@dataclass
class GameRecord:
    game_id: int
    sf_depth: int
    model_color: str          # "white" or "black"
    outcome: str              # "win" / "draw" / "loss" from model perspective
    termination: str          # e.g. "checkmate", "stalemate", "fifty_moves", etc.
    pgn: str                  # full PGN string
    moves: list[MoveRecord] = field(default_factory=list)


def _parse_uci(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).lower() if m else None


def _query_model(sampling_client, renderer, fen: str, max_tokens: int) -> tuple[str, str | None]:
    """Query model. Returns (raw_response_text, parsed_uci_or_None)."""
    messages = [{"role": "user", "content": _USER_PREFIX + fen + _USER_SUFFIX}]
    prompt = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()
    params = tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=stop)
    response = sampling_client.sample(prompt, num_samples=1, sampling_params=params).result()
    if not response.sequences:
        return "", None
    tokens = response.sequences[0].tokens
    text = renderer.tokenizer.decode(tokens, skip_special_tokens=True)
    return text, _parse_uci(text)


def _play_game(
    sampling_client,
    renderer,
    sf_engine: chess.engine.SimpleEngine,
    game_id: int,
    sf_depth: int,
    model_is_white: bool,
    max_tokens: int,
) -> GameRecord:
    """Play one full game and return a detailed GameRecord."""
    board = chess.Board()
    move_records: list[MoveRecord] = []

    while not board.is_game_over(claim_draw=True):
        ply = board.fullmove_number * 2 - (2 if board.turn == chess.WHITE else 1)
        fen_before = board.fen()
        model_turn = (board.turn == chess.WHITE) == model_is_white

        if model_turn:
            raw, uci_parsed = _query_model(sampling_client, renderer, fen_before, max_tokens)
            move = None
            legal = False
            if uci_parsed:
                try:
                    candidate = chess.Move.from_uci(uci_parsed)
                    if candidate in board.legal_moves:
                        move = candidate
                        legal = True
                except ValueError:
                    pass
            if move is None:
                move = next(iter(board.legal_moves))  # fallback

            move_records.append(MoveRecord(
                ply=ply, side="model", fen_before=fen_before,
                raw_response=raw, uci_parsed=uci_parsed,
                uci_played=move.uci(), legal=legal,
            ))
        else:
            result = sf_engine.play(board, chess.engine.Limit(depth=sf_depth))
            move = result.move or next(iter(board.legal_moves))
            move_records.append(MoveRecord(
                ply=ply, side="stockfish", fen_before=fen_before,
                raw_response="", uci_parsed=move.uci(),
                uci_played=move.uci(), legal=True,
            ))

        board.push(move)

    outcome_obj = board.outcome(claim_draw=True)
    if outcome_obj is None or outcome_obj.winner is None:
        outcome = "draw"
    else:
        model_color = chess.WHITE if model_is_white else chess.BLACK
        outcome = "win" if outcome_obj.winner == model_color else "loss"

    termination = outcome_obj.termination.name.lower() if outcome_obj else "unknown"

    # Build PGN
    pgn_board = chess.Board()
    san_moves = []
    for rec in move_records:
        m = chess.Move.from_uci(rec.uci_played)
        san_moves.append(pgn_board.san(m))
        pgn_board.push(m)
    pgn_tokens = []
    for i, san in enumerate(san_moves):
        if i % 2 == 0:
            pgn_tokens.append(f"{i // 2 + 1}.")
        pgn_tokens.append(san)
    pgn = " ".join(pgn_tokens)

    return GameRecord(
        game_id=game_id,
        sf_depth=sf_depth,
        model_color="white" if model_is_white else "black",
        outcome=outcome,
        termination=termination,
        pgn=pgn,
        moves=move_records,
    )


def _run_depth(
    depth: int,
    games_per_depth: int,
    sampling_client,
    renderer,
    max_tokens: int,
    game_id_offset: int,
    bar: tqdm,
    results: dict,
    log_lock: threading.Lock,
    log_f,
) -> list[GameRecord]:
    """Run all games for one depth level with a dedicated SF engine."""
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    records = []
    try:
        for game_idx in range(games_per_depth):
            model_is_white = game_idx % 2 == 0
            game_id = game_id_offset + game_idx
            record = _play_game(
                sampling_client, renderer, sf_engine,
                game_id, depth, model_is_white, max_tokens,
            )
            records.append(record)

            with log_lock:
                results[depth][record.outcome] += 1
                log_f.write(json.dumps({
                    **{k: v for k, v in asdict(record).items() if k != "moves"},
                    "moves": [asdict(m) for m in record.moves],
                }) + "\n")
                log_f.flush()

                illegal_count = sum(1 for m in record.moves if m.side == "model" and not m.legal)
                tqdm.write(
                    f"  [D{depth:02d} G{game_idx+1}] {record.model_color:5s} | "
                    f"{record.outcome:4s} | {record.termination:15s} | "
                    f"{len(record.moves):3d} plies | {illegal_count} illegal | "
                    f"{record.pgn[:60]}{'…' if len(record.pgn) > 60 else ''}"
                )
                bar.update(1)
    finally:
        sf_engine.quit()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint vs Stockfish")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Tinker sampler_weights path, e.g. tinker://<id>:train:0/sampler_weights/final",
    )
    parser.add_argument("--games-per-depth", type=int, default=20)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--tinker-base-url", type=str, default=None)
    parser.add_argument(
        "--game-log", type=str, default=None,
        help="Path to write per-game JSONL log. Defaults to outputs/games_<checkpoint_id>.jsonl",
    )
    args = parser.parse_args()

    ckpt_id = args.checkpoint.replace("/", "_").replace(":", "_")[-40:]
    log_path = Path(args.game_log or f"outputs/games_{ckpt_id}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    service_client = tinker.ServiceClient(base_url=args.tinker_base_url)
    sampling_client = service_client.create_sampling_client(model_path=args.checkpoint)

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer("kimi_k25", tokenizer)

    results: dict[int, dict[str, int]] = defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0})
    total_games = len(args.depths) * args.games_per_depth
    log_lock = threading.Lock()

    print(f"Running {total_games} games across {len(args.depths)} depths in parallel...")
    print(f"Game log: {log_path}")

    with log_path.open("w") as log_f, tqdm(total=total_games, unit="game") as bar:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(args.depths)) as executor:
            for i, depth in enumerate(args.depths):
                game_id_offset = i * args.games_per_depth
                fut = executor.submit(
                    _run_depth,
                    depth, args.games_per_depth,
                    sampling_client, renderer, args.max_tokens,
                    game_id_offset, bar, results, log_lock, log_f,
                )
                futures[fut] = depth

            for fut in as_completed(futures):
                depth = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    tqdm.write(f"  [D{depth:02d}] ERROR: {e}")

    print(f"\nGame log written to: {log_path}")
    print(f"\n{'Depth':>6} {'Win':>6} {'Draw':>6} {'Loss':>6} {'Win%':>7}")
    print("-" * 38)
    for depth in sorted(results):
        r = results[depth]
        total = sum(r.values())
        win_pct = 100 * r["win"] / total if total else 0
        print(f"{depth:>6} {r['win']:>6} {r['draw']:>6} {r['loss']:>6} {win_pct:>6.1f}%")


if __name__ == "__main__":
    main()

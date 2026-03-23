"""Evaluate a GRPO/SFT checkpoint by playing full games against Stockfish.

Plays N games per Stockfish depth, alternating colours, with all depths
running in parallel. Reports win/draw/loss rates per depth.

Usage:
    uv run python src/eval_vs_stockfish.py \\
        --checkpoint tinker://<id>:train:0/sampler_weights/000050 \\
        --games-per-depth 10 \\
        --depths 1 2 3 4 5
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.engine
import tinker
from rllm.parser.chat_template_parser import ChatTemplateParser
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from prompt import format_prompt

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
_MOVE_RE = re.compile(r"<move>\s*([^<]+?)\s*</move>")


@dataclass
class MoveRecord:
    ply: int
    side: str
    fen_before: str
    raw_response: str
    san_parsed: str | None
    uci_played: str
    legal: bool


@dataclass
class GameRecord:
    game_id: int
    sf_depth: int
    model_color: str
    outcome: str
    termination: str
    pgn: str
    moves: list[MoveRecord] = field(default_factory=list)


def _parse_move(text: str) -> str | None:
    """Extract SAN/UCI from <move>...</move> tag."""
    m = _MOVE_RE.search(text)
    return m.group(1).strip() if m else None


def _query_model(
    sampling_client: tinker.SamplingClient,
    parser: ChatTemplateParser,
    tokenizer,
    fen: str,
    max_tokens: int,
) -> tuple[str, str | None]:
    """Query the model for a move. Returns (raw_text, parsed_san_or_None)."""
    messages = [{"role": "user", "content": format_prompt(fen)}]
    prompt_str = parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    params = tinker.types.SamplingParams(
        max_tokens=max_tokens, temperature=0.0, stop=parser.stop_sequences
    )
    response = sampling_client.sample(
        tinker.types.ModelInput.from_ints(prompt_ids), num_samples=1, sampling_params=params
    ).result()
    if not response.sequences:
        return "", None
    tokens = response.sequences[0].tokens
    parsed = parser.parse_completion(tokens)
    text = parsed.get("content") or tokenizer.decode(tokens, skip_special_tokens=True)
    return text, _parse_move(text)


def _play_game(
    sampling_client: tinker.SamplingClient,
    parser: ChatTemplateParser,
    tokenizer,
    sf_engine: chess.engine.SimpleEngine,
    game_id: int,
    sf_depth: int,
    model_is_white: bool,
    max_tokens: int,
) -> GameRecord:
    """Play one full game and return a GameRecord."""
    board = chess.Board()
    move_records: list[MoveRecord] = []

    while not board.is_game_over(claim_draw=True):
        ply = board.fullmove_number * 2 - (2 if board.turn == chess.WHITE else 1)
        fen_before = board.fen()
        model_turn = (board.turn == chess.WHITE) == model_is_white

        if model_turn:
            raw, san_parsed = _query_model(sampling_client, parser, tokenizer, fen_before, max_tokens)
            san_map = {board.san(mv): mv for mv in board.legal_moves}
            uci_map = {mv.uci(): mv for mv in board.legal_moves}
            move = (
                san_map.get(san_parsed or "")
                or san_map.get((san_parsed or "").rstrip("+#"))
                or uci_map.get((san_parsed or "").lower())
                or next(iter(board.legal_moves))
            )
            legal = move != next(iter(board.legal_moves)) or san_parsed in san_map or san_parsed in uci_map
            move_records.append(MoveRecord(
                ply=ply, side="model", fen_before=fen_before,
                raw_response=raw, san_parsed=san_parsed,
                uci_played=move.uci(), legal=san_parsed is not None and (san_parsed in san_map or san_parsed in uci_map),
            ))
        else:
            result = sf_engine.play(board, chess.engine.Limit(depth=sf_depth))
            move = result.move or next(iter(board.legal_moves))
            move_records.append(MoveRecord(
                ply=ply, side="stockfish", fen_before=fen_before,
                raw_response="", san_parsed=move.uci(),
                uci_played=move.uci(), legal=True,
            ))

        board.push(move)

    outcome_obj = board.outcome(claim_draw=True)
    if outcome_obj is None or outcome_obj.winner is None:
        outcome = "draw"
    else:
        outcome = "win" if outcome_obj.winner == (chess.WHITE if model_is_white else chess.BLACK) else "loss"

    termination = outcome_obj.termination.name.lower() if outcome_obj else "unknown"

    pgn_board = chess.Board()
    san_moves = []
    for rec in move_records:
        m = chess.Move.from_uci(rec.uci_played)
        san_moves.append(pgn_board.san(m))
        pgn_board.push(m)
    pgn_tokens: list[str] = []
    for i, san in enumerate(san_moves):
        if i % 2 == 0:
            pgn_tokens.append(f"{i // 2 + 1}.")
        pgn_tokens.append(san)
    pgn = " ".join(pgn_tokens)

    return GameRecord(
        game_id=game_id, sf_depth=sf_depth,
        model_color="white" if model_is_white else "black",
        outcome=outcome, termination=termination, pgn=pgn, moves=move_records,
    )


def _run_depth(
    depth: int,
    games_per_depth: int,
    sampling_client: tinker.SamplingClient,
    parser: ChatTemplateParser,
    tokenizer,
    max_tokens: int,
    game_id_offset: int,
    bar: tqdm,
    results: dict,
    log_lock: threading.Lock,
    log_f,
) -> list[GameRecord]:
    """Run all games for one Stockfish depth with a dedicated engine instance."""
    sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    records = []
    for game_idx in range(games_per_depth):
        record = _play_game(
            sampling_client, parser, tokenizer, sf_engine,
            game_id_offset + game_idx, depth, game_idx % 2 == 0, max_tokens,
        )
        records.append(record)
        with log_lock:
            results[depth][record.outcome] += 1
            log_f.write(json.dumps({
                **{k: v for k, v in asdict(record).items() if k != "moves"},
                "moves": [asdict(m) for m in record.moves],
            }) + "\n")
            log_f.flush()
            illegal = sum(1 for m in record.moves if m.side == "model" and not m.legal)
            tqdm.write(
                f"  [D{depth:02d} G{game_idx+1}] {record.model_color:5s} | "
                f"{record.outcome:4s} | {record.termination:15s} | "
                f"{len(record.moves):3d} plies | {illegal} illegal"
            )
            bar.update(1)
    sf_engine.quit()
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games-per-depth", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tinker-base-url", type=str, default=None)
    ap.add_argument("--game-log", type=str, default=None)
    args = ap.parse_args()

    ckpt_id = args.checkpoint.replace("/", "_").replace(":", "_")[-40:]
    log_path = Path(args.game_log or f"outputs/games_{ckpt_id}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    sampling_client = tinker.ServiceClient(base_url=args.tinker_base_url).create_sampling_client(
        model_path=args.checkpoint
    )
    tokenizer = get_tokenizer(MODEL_NAME)
    parser = ChatTemplateParser.get_parser(tokenizer, disable_thinking=False)

    results: dict[int, dict[str, int]] = defaultdict(lambda: {"win": 0, "draw": 0, "loss": 0})
    log_lock = threading.Lock()
    total_games = len(args.depths) * args.games_per_depth
    print(f"Running {total_games} games across depths {args.depths} | log: {log_path}")

    with log_path.open("w") as log_f, tqdm(total=total_games, unit="game") as bar:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(args.depths)) as executor:
            for i, depth in enumerate(args.depths):
                futures[executor.submit(
                    _run_depth, depth, args.games_per_depth,
                    sampling_client, parser, tokenizer, args.max_tokens,
                    i * args.games_per_depth, bar, results, log_lock, log_f,
                )] = depth
            for fut in as_completed(futures):
                depth = futures[fut]
                exc = fut.exception()
                if exc:
                    tqdm.write(f"  [D{depth:02d}] FAILED: {exc}")

    print(f"\n{'Depth':>6} {'Win':>6} {'Draw':>6} {'Loss':>6} {'Win%':>7}")
    print("-" * 38)
    for depth in sorted(results):
        r = results[depth]
        total = sum(r.values())
        print(f"{depth:>6} {r['win']:>6} {r['draw']:>6} {r['loss']:>6} {100*r['win']/total:>6.1f}%")


if __name__ == "__main__":
    main()

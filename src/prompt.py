"""Shared chess prompt formatting utilities."""
from __future__ import annotations

import chess


_PIECE_SYMBOLS = {
    chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b",
    chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k",
}


def board_text(board: chess.Board) -> str:
    """Render a board as an 8x8 ASCII grid.

    Args:
        board: Position to render.

    Returns:
        8-line string with ranks 8→1, pieces as letters, empty squares as '.'.
    """
    rows = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                row.append(".")
            else:
                sym = _PIECE_SYMBOLS[piece.piece_type]
                row.append(sym.upper() if piece.color == chess.WHITE else sym)
        rows.append(" ".join(row))
    return "\n".join(rows)


def format_prompt(fen: str) -> str:
    """Build the user message for a chess position.

    Args:
        fen: FEN string of the position.

    Returns:
        Formatted user message string with board diagram, FEN, side to move,
        legal moves, and response instruction.
    """
    board = chess.Board(fen)
    side = "White" if board.turn == chess.WHITE else "Black"
    legal = sorted(board.san(mv) for mv in board.legal_moves)
    legal_str = ", ".join(legal)
    return (
        f"=== Chess ===\n"
        f"{board_text(board)}\n\n"
        f"FEN: {fen}\n"
        f"Side to move: {side}\n"
        f"Legal moves ({len(legal)}): {legal_str}\n\n"
        f"Respond with your move in <move>SAN</move> format."
    )

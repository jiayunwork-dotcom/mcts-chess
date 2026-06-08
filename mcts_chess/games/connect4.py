from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcts_chess.games.base import BaseGame

ROWS: int = 6
COLS: int = 7
WIN_LENGTH: int = 4


class Connect4(BaseGame):
    def __init__(
        self,
        board: NDArray[np.int8] | None = None,
        current_player: int = 1,
    ) -> None:
        if board is not None:
            self._board: NDArray[np.int8] = board.copy()
        else:
            self._board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._current_player: int = current_player

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def action_size(self) -> int:
        return COLS

    def get_legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self._board[0, c] == 0]

    def make_move(self, action: int) -> Connect4:
        new_board = self._board.copy()
        for row in range(ROWS - 1, -1, -1):
            if new_board[row, action] == 0:
                new_board[row, action] = np.int8(self._current_player)
                break
        return Connect4(board=new_board, current_player=-self._current_player)

    def _check_four_in_a_row(self, player: int) -> bool:
        piece = np.int8(player)
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(self._board[r, c + i] == piece for i in range(WIN_LENGTH)):
                    return True
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS):
                if all(self._board[r + i, c] == piece for i in range(WIN_LENGTH)):
                    return True
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(self._board[r + i, c + i] == piece for i in range(WIN_LENGTH)):
                    return True
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(self._board[r - i, c + i] == piece for i in range(WIN_LENGTH)):
                    return True
        return False

    def is_game_over(self) -> bool:
        if self._check_four_in_a_row(1) or self._check_four_in_a_row(-1):
            return True
        return bool(np.all(self._board != 0))

    def get_result(self) -> float:
        if self._check_four_in_a_row(self._current_player):
            return 1.0
        if self._check_four_in_a_row(-self._current_player):
            return -1.0
        return 0.0

    def get_current_player(self) -> int:
        return self._current_player

    def get_state_hash(self) -> int:
        return hash(self._board.tobytes())

    def get_canonical_form(self) -> np.ndarray:
        return self._board * np.int8(self._current_player)

    def get_feature_planes(self) -> np.ndarray:
        current_mask = (self._board == self._current_player).astype(np.float32)
        opponent_mask = (self._board == -self._current_player).astype(np.float32)
        indicator = np.full((ROWS, COLS), 1.0 if self._current_player == 1 else 0.0, dtype=np.float32)
        feature = np.stack([current_mask, opponent_mask, indicator], axis=0)
        return feature

    def copy(self) -> Connect4:
        return Connect4(board=self._board, current_player=self._current_player)

    def to_string(self) -> str:
        lines: list[str] = []
        for r in range(ROWS):
            row_str = " ".join(
                {1: "X", -1: "O", 0: "."}[int(self._board[r, c])]
                for c in range(COLS)
            )
            lines.append(row_str)
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)

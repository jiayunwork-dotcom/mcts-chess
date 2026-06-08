from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcts_chess.games.base import BaseGame

BOARD_SIZE = 8
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1),
]


class Othello(BaseGame):
    def __init__(
        self,
        board: NDArray[np.int8] | None = None,
        current_player: int = 1,
    ) -> None:
        if board is not None:
            self._board = board.copy()
        else:
            self._board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            self._board[3, 3] = -1
            self._board[3, 4] = 1
            self._board[4, 3] = 1
            self._board[4, 4] = -1
        self._current_player: int = current_player

    @property
    def board(self) -> NDArray[np.int8]:
        return self._board

    @property
    def action_size(self) -> int:
        return BOARD_SIZE * BOARD_SIZE

    def get_current_player(self) -> int:
        return self._current_player

    def _flips_in_direction(
        self, row: int, col: int, dr: int, dc: int, player: int
    ) -> list[tuple[int, int]]:
        opponent = -player
        flips: list[tuple[int, int]] = []
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self._board[r, c] == opponent:
                flips.append((r, c))
            elif self._board[r, c] == player:
                return flips
            else:
                break
            r += dr
            c += dc
        return []

    def _get_flips(self, row: int, col: int, player: int) -> list[tuple[int, int]]:
        if self._board[row, col] != 0:
            return []
        all_flips: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            all_flips.extend(self._flips_in_direction(row, col, dr, dc, player))
        return all_flips

    def _get_flips_on_board(
        self, row: int, col: int, player: int, board: NDArray[np.int8]
    ) -> list[tuple[int, int]]:
        if board[row, col] != 0:
            return []
        all_flips: list[tuple[int, int]] = []
        opponent = -player
        for dr, dc in DIRECTIONS:
            flips: list[tuple[int, int]] = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if board[r, c] == opponent:
                    flips.append((r, c))
                elif board[r, c] == player:
                    all_flips.extend(flips)
                    break
                else:
                    break
                r += dr
                c += dc
        return all_flips

    def _has_legal_move_on_board(
        self, player: int, board: NDArray[np.int8]
    ) -> bool:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == 0 and self._get_flips_on_board(r, c, player, board):
                    return True
        return False

    def get_legal_moves(self) -> list[int]:
        moves: list[int] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._board[r, c] == 0 and self._get_flips(r, c, self._current_player):
                    moves.append(r * BOARD_SIZE + c)
        return moves

    def make_move(self, action: int) -> Othello:
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        flips = self._get_flips(row, col, self._current_player)
        new_board = self._board.copy()
        new_board[row, col] = self._current_player
        for fr, fc in flips:
            new_board[fr, fc] = self._current_player
        opponent = -self._current_player
        next_player = opponent if self._has_legal_move_on_board(opponent, new_board) else self._current_player
        return Othello(new_board, next_player)

    def is_game_over(self) -> bool:
        if self._has_legal_move_on_board(self._current_player, self._board):
            return False
        if self._has_legal_move_on_board(-self._current_player, self._board):
            return False
        return True

    def get_result(self) -> float:
        black_count = int(np.sum(self._board == 1))
        white_count = int(np.sum(self._board == -1))
        if self._current_player == 1:
            player_count = black_count
            opponent_count = white_count
        else:
            player_count = white_count
            opponent_count = black_count
        if player_count > opponent_count:
            return 1.0
        if player_count < opponent_count:
            return -1.0
        return 0.0

    def get_state_hash(self) -> int:
        return hash(self._board.tobytes() + bytes([self._current_player]))

    def get_canonical_form(self) -> np.ndarray:
        return self._board * np.int8(self._current_player)

    def get_feature_planes(self) -> NDArray[np.float32]:
        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self._board == self._current_player).astype(np.float32)
        planes[1] = (self._board == -self._current_player).astype(np.float32)
        planes[2] = np.full(
            (BOARD_SIZE, BOARD_SIZE),
            1.0 if self._current_player == 1 else 0.0,
            dtype=np.float32,
        )
        return planes

    def copy(self) -> Othello:
        return Othello(self._board.copy(), self._current_player)

    def to_string(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        lines = []
        for row in self._board:
            lines.append(" ".join(symbols[int(v)] for v in row))
        return "\n".join(lines)

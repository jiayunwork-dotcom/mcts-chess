from __future__ import annotations

import numpy as np

from mcts_chess.games.base import BaseGame

BOARD_SIZE = 15
WIN_LENGTH = 5
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Gomoku(BaseGame):

    def __init__(
        self,
        board: np.ndarray | None = None,
        current_player: int = 1,
        last_move: int | None = None,
    ) -> None:
        if board is not None:
            self._board = board
        else:
            self._board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._current_player: int = current_player
        self._last_move: int | None = last_move
        self._game_over: bool | None = None
        self._winner: int | None = None

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def action_size(self) -> int:
        return BOARD_SIZE * BOARD_SIZE

    def get_current_player(self) -> int:
        return self._current_player

    def get_legal_moves(self) -> list[int]:
        if np.count_nonzero(self._board) == 0:
            return [BOARD_SIZE // 2 * BOARD_SIZE + BOARD_SIZE // 2]

        rows, cols = np.where(self._board != 0)
        if rows.size == 0:
            return [BOARD_SIZE // 2 * BOARD_SIZE + BOARD_SIZE // 2]

        occupied = set(zip(rows.tolist(), cols.tolist()))
        candidates: list[int] = []
        seen: set[tuple[int, int]] = set()

        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < BOARD_SIZE
                        and 0 <= nc < BOARD_SIZE
                        and self._board[nr, nc] == 0
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        candidates.append(nr * BOARD_SIZE + nc)

        return candidates

    def make_move(self, action: int) -> Gomoku:
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        new_board = self._board.copy()
        new_board[row, col] = self._current_player
        return Gomoku(
            board=new_board,
            current_player=-self._current_player,
            last_move=action,
        )

    def is_game_over(self) -> bool:
        if self._game_over is not None:
            return self._game_over

        if self._last_move is None:
            self._game_over = False
            return False

        row = self._last_move // BOARD_SIZE
        col = self._last_move % BOARD_SIZE
        player = self._board[row, col]

        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while (
                    0 <= r < BOARD_SIZE
                    and 0 <= c < BOARD_SIZE
                    and self._board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= WIN_LENGTH:
                self._winner = player
                self._game_over = True
                return True

        if np.count_nonzero(self._board) == BOARD_SIZE * BOARD_SIZE:
            self._game_over = True
            return True

        self._game_over = False
        return False

    def get_result(self) -> float:
        if not self.is_game_over():
            raise RuntimeError("Game is not over yet")

        if self._winner is not None:
            return 1.0 if self._winner == self._current_player else -1.0

        return 0.0

    def get_state_hash(self) -> int:
        return hash(self._board.tobytes())

    def get_canonical_form(self) -> np.ndarray:
        return self._board * self._current_player

    def get_feature_planes(self) -> np.ndarray:
        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self._board == self._current_player).astype(np.float32)
        planes[1] = (self._board == -self._current_player).astype(np.float32)
        planes[2] = (np.full((BOARD_SIZE, BOARD_SIZE), 1 if self._current_player == 1 else 0, dtype=np.float32))
        return planes

    def copy(self) -> Gomoku:
        return Gomoku(
            board=self._board.copy(),
            current_player=self._current_player,
            last_move=self._last_move,
        )

    def to_string(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        lines = []
        for row in self._board:
            lines.append(" ".join(symbols[v] for v in row))
        return "\n".join(lines)

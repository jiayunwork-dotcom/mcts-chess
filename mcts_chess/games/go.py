from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from mcts_chess.games.base import BaseGame

BOARD_SIZE = 9
ACTION_SIZE = 82
KOMI = 3.75
HISTORY_LENGTH = 8
FEATURE_PLANES = 19


class GoGame(BaseGame):
    def __init__(
        self,
        board: NDArray[np.int8] | None = None,
        current_player: int = 1,
        consecutive_passes: int = 0,
        ko_point: tuple[int, int] | None = None,
        move_history: list[NDArray[np.int8]] | None = None,
    ) -> None:
        if board is not None:
            self._board = board.copy()
        else:
            self._board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._current_player: int = current_player
        self._consecutive_passes: int = consecutive_passes
        self._ko_point: tuple[int, int] | None = ko_point
        self._move_history: list[NDArray[np.int8]] = (
            [b.copy() for b in move_history] if move_history else []
        )

    @property
    def board(self) -> NDArray[np.int8]:
        return self._board

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    def get_current_player(self) -> int:
        return self._current_player

    def _get_group(
        self, row: int, col: int, board: NDArray[np.int8]
    ) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        color = board[row, col]
        if color == 0:
            return [], set()
        group: list[tuple[int, int]] = []
        liberties: set[tuple[int, int]] = set()
        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        queue.append((row, col))
        visited.add((row, col))
        while queue:
            r, c = queue.popleft()
            group.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) in visited:
                        continue
                    visited.add((nr, nc))
                    if board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif board[nr, nc] == color:
                        queue.append((nr, nc))
        return group, liberties

    def _is_suicide(self, row: int, col: int, board: NDArray[np.int8], player: int) -> bool:
        if board[row, col] != 0:
            return True
        test_board = board.copy()
        test_board[row, col] = np.int8(player)
        opponent = -player
        captures_opponent = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if test_board[nr, nc] == opponent:
                    _, libs = self._get_group(nr, nc, test_board)
                    if len(libs) == 0:
                        captures_opponent = True
                        break
        if captures_opponent:
            return False
        _, own_libs = self._get_group(row, col, test_board)
        return len(own_libs) == 0

    def get_legal_moves(self) -> list[int]:
        moves: list[int] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._board[r, c] != 0:
                    continue
                if self._ko_point == (r, c):
                    continue
                if self._is_suicide(r, c, self._board, self._current_player):
                    continue
                moves.append(r * BOARD_SIZE + c)
        moves.append(ACTION_SIZE - 1)
        return moves

    def make_move(self, action: int) -> GoGame:
        new_history = [b.copy() for b in self._move_history]
        new_history.append(self._board.copy())

        if action == ACTION_SIZE - 1:
            return GoGame(
                board=self._board.copy(),
                current_player=-self._current_player,
                consecutive_passes=self._consecutive_passes + 1,
                ko_point=None,
                move_history=new_history,
            )

        new_board = self._board.copy()
        new_board[action // BOARD_SIZE, action % BOARD_SIZE] = np.int8(self._current_player)
        opponent = -self._current_player
        total_captured: list[tuple[int, int]] = []
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if new_board[nr, nc] == opponent:
                    group, libs = self._get_group(nr, nc, new_board)
                    if len(libs) == 0:
                        total_captured.extend(group)
                        for gr, gc in group:
                            new_board[gr, gc] = 0

        new_ko: tuple[int, int] | None = None
        if len(total_captured) == 1:
            cr, cc = total_captured[0]
            _, libs = self._get_group(row, col, new_board)
            if len(libs) == 1 and (cr, cc) in libs:
                new_ko = (cr, cc)

        return GoGame(
            board=new_board,
            current_player=-self._current_player,
            consecutive_passes=0,
            ko_point=new_ko,
            move_history=new_history,
        )

    def is_game_over(self) -> bool:
        return self._consecutive_passes >= 2

    def get_result(self) -> float:
        visited: set[tuple[int, int]] = set()
        black_territory = 0
        white_territory = 0
        black_stones = 0
        white_stones = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._board[r, c] == 1:
                    black_stones += 1
                elif self._board[r, c] == -1:
                    white_stones += 1
                elif (r, c) not in visited:
                    region: list[tuple[int, int]] = []
                    borders: set[int] = set()
                    queue: deque[tuple[int, int]] = deque()
                    queue.append((r, c))
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        region.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if self._board[nr, nc] != 0:
                                    borders.add(int(self._board[nr, nc]))
                                elif (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                    if borders == {1}:
                        black_territory += len(region)
                    elif borders == {-1}:
                        white_territory += len(region)
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory + KOMI
        winner = 1 if black_score > white_score else (-1 if white_score > black_score else 0)
        if winner == 0:
            return 0.0
        return 1.0 if winner == self._current_player else -1.0

    def get_state_hash(self) -> int:
        return hash(self._board.tobytes() + bytes([self._current_player]))

    def get_canonical_form(self) -> np.ndarray:
        return self._board * np.int8(self._current_player)

    def get_feature_planes(self) -> NDArray[np.float32]:
        planes = np.zeros((FEATURE_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self._board == self._current_player).astype(np.float32)
        planes[1] = (self._board == -self._current_player).astype(np.float32)
        planes[2] = np.full(
            (BOARD_SIZE, BOARD_SIZE),
            1.0 if self._current_player == 1 else 0.0,
            dtype=np.float32,
        )
        history = self._move_history
        for i in range(HISTORY_LENGTH):
            idx = len(history) - 1 - i
            if idx >= 0:
                past_board = history[idx]
                past_player = self._current_player * ((-1) ** (i + 1))
                planes[3 + i * 2] = (past_board == past_player).astype(np.float32)
                planes[3 + i * 2 + 1] = (
                    past_board == -past_player
                ).astype(np.float32)
        return planes

    def copy(self) -> GoGame:
        return GoGame(
            board=self._board.copy(),
            current_player=self._current_player,
            consecutive_passes=self._consecutive_passes,
            ko_point=self._ko_point,
            move_history=self._move_history,
        )

    def to_string(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        lines = []
        for row in self._board:
            lines.append(" ".join(symbols[int(v)] for v in row))
        return "\n".join(lines)

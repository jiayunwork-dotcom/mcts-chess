from abc import ABC, abstractmethod

import numpy as np


class BaseGame(ABC):

    @property
    @abstractmethod
    def board(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def action_size(self) -> int:
        ...

    @abstractmethod
    def get_legal_moves(self) -> list:
        ...

    @abstractmethod
    def make_move(self, action: int) -> "BaseGame":
        ...

    @abstractmethod
    def is_game_over(self) -> bool:
        ...

    @abstractmethod
    def get_result(self) -> float:
        ...

    @abstractmethod
    def get_current_player(self) -> int:
        ...

    @abstractmethod
    def get_state_hash(self) -> int:
        ...

    @abstractmethod
    def get_canonical_form(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_feature_planes(self) -> np.ndarray:
        ...

    @abstractmethod
    def copy(self) -> "BaseGame":
        ...

    @abstractmethod
    def to_string(self) -> str:
        ...

from __future__ import annotations

import pickle
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 500000) -> None:
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        self._buffer.append((state, policy, value))

    def sample(
        self, batch_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        states = np.array([self._buffer[i][0] for i in indices])
        policies = np.array([self._buffer[i][1] for i in indices])
        values = np.array([self._buffer[i][2] for i in indices])
        return states, policies, values

    def __len__(self) -> int:
        return len(self._buffer)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(list(self._buffer), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._buffer = deque(data, maxlen=self._buffer.maxlen)

from __future__ import annotations

import multiprocessing
from typing import Callable

import numpy as np
import torch

from mcts_chess.games import Gomoku, Connect4, Othello, GoGame
from mcts_chess.mcts.engine import MCTSEngine
from mcts_chess.network.model import AlphaZeroNet
from mcts_chess.selfplay.buffer import ReplayBuffer


def _play_game_worker(args: tuple) -> list[tuple[np.ndarray, np.ndarray, float]]:
    game_class, model_state_dict, model_config, num_iterations, c_puct, temperature_threshold, device = args
    model = AlphaZeroNet(**model_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    game = game_class()
    engine = MCTSEngine(
        num_iterations=num_iterations,
        c_puct=c_puct,
        use_puct=True,
        neural_net=model,
        temperature_threshold=temperature_threshold,
    )

    samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    move_count = 0

    while not game.is_game_over():
        canonical = game.get_canonical_form()
        feature = game.get_feature_planes()
        best_action, policy = engine.search(game)

        samples.append((feature.copy(), policy.copy(), 0.0))

        game = game.make_move(best_action)
        move_count += 1

    result = game.get_result()

    augmented: list[tuple[np.ndarray, np.ndarray, float]] = []
    num_samples = len(samples)
    for i, (state, pol, _) in enumerate(samples):
        perspective = 1.0 if i % 2 == 0 else -1.0
        value = result * perspective

        if issubclass(game_class, (Gomoku, GoGame)):
            board_size = state.shape[-1]
            augmented_pairs = _augment_gomoku(state, pol, board_size)
        elif issubclass(game_class, Connect4):
            augmented_pairs = _augment_connect4(state, pol)
        elif issubclass(game_class, Othello):
            board_size = state.shape[-1]
            augmented_pairs = _augment_go(state, pol, board_size)
        else:
            augmented_pairs = [(state, pol)]

        for aug_state, aug_pol in augmented_pairs:
            augmented.append((aug_state, aug_pol, value))

    return augmented


def _augment_gomoku(
    state: np.ndarray, policy: np.ndarray, board_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray]] = []
    policy_2d = policy.reshape(board_size, board_size)

    for k in range(4):
        rot_state = np.rot90(state, k=k, axes=(-2, -1))
        rot_policy = np.rot90(policy_2d, k=k)
        results.append((rot_state.copy(), rot_policy.reshape(-1).copy()))

        flip_state = np.flip(rot_state, axis=-1)
        flip_policy = np.fliplr(rot_policy)
        results.append((flip_state.copy(), flip_policy.reshape(-1).copy()))

    return results


def _augment_connect4(
    state: np.ndarray, policy: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray]] = [(state.copy(), policy.copy())]
    flip_state = np.flip(state, axis=-1)
    flip_policy = np.flip(policy)
    results.append((flip_state.copy(), flip_policy.copy()))
    return results


def _augment_go(
    state: np.ndarray, policy: np.ndarray, board_size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray]] = []
    policy_2d = policy.reshape(board_size, board_size)

    for k in range(4):
        rot_state = np.rot90(state, k=k, axes=(-2, -1))
        rot_policy = np.rot90(policy_2d, k=k)
        results.append((rot_state.copy(), rot_policy.reshape(-1).copy()))

        flip_state = np.flip(rot_state, axis=-1)
        flip_policy = np.fliplr(rot_policy)
        results.append((flip_state.copy(), flip_policy.reshape(-1).copy()))

    return results


class SelfPlayPipeline:
    def __init__(
        self,
        game_class: type,
        model: AlphaZeroNet,
        buffer: ReplayBuffer,
        num_parallel: int = 4,
        num_iterations: int = 400,
        c_puct: float = 1.5,
        temperature_threshold: int = 30,
        device: str = "cpu",
    ) -> None:
        self.game_class = game_class
        self.model = model
        self.buffer = buffer
        self.num_parallel = num_parallel
        self.num_iterations = num_iterations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.device = device

    def play_game(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        game = self.game_class()
        engine = MCTSEngine(
            num_iterations=self.num_iterations,
            c_puct=self.c_puct,
            use_puct=True,
            neural_net=self.model,
            temperature_threshold=self.temperature_threshold,
        )

        samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        move_count = 0

        while not game.is_game_over():
            canonical = game.get_canonical_form()
            feature = game.get_feature_planes()
            best_action, policy = engine.search(game)

            samples.append((feature.copy(), policy.copy(), 0.0))

            game = game.make_move(best_action)
            move_count += 1

        result = game.get_result()

        augmented: list[tuple[np.ndarray, np.ndarray, float]] = []
        num_samples = len(samples)
        for i, (state, pol, _) in enumerate(samples):
            perspective = 1.0 if i % 2 == 0 else -1.0
            value = result * perspective

            if issubclass(self.game_class, (Gomoku, GoGame)):
                board_size = state.shape[-1]
                augmented_pairs = _augment_gomoku(state, pol, board_size)
            elif issubclass(self.game_class, Connect4):
                augmented_pairs = _augment_connect4(state, pol)
            elif issubclass(self.game_class, Othello):
                board_size = state.shape[-1]
                augmented_pairs = _augment_go(state, pol, board_size)
            else:
                augmented_pairs = [(state, pol)]

            for aug_state, aug_pol in augmented_pairs:
                augmented.append((aug_state, aug_pol, value))

        return augmented

    def generate(self, num_games: int = 100) -> None:
        model_config = {
            "input_channels": self.model.conv_in.in_channels,
            "board_size": self.model.board_size,
            "action_size": self.model.action_size,
            "num_res_blocks": len(self.model.res_blocks),
            "channels": self.model.conv_in.out_channels,
        }
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if self.num_parallel > 1:
            args_list = [
                (
                    self.game_class,
                    model_state_dict,
                    model_config,
                    self.num_iterations,
                    self.c_puct,
                    self.temperature_threshold,
                    self.device,
                )
                for _ in range(num_games)
            ]
            with multiprocessing.Pool(processes=self.num_parallel) as pool:
                all_results = pool.map(_play_game_worker, args_list)
            for game_data in all_results:
                for state, policy, value in game_data:
                    self.buffer.add(state, policy, value)
        else:
            for _ in range(num_games):
                game_data = self.play_game()
                for state, policy, value in game_data:
                    self.buffer.add(state, policy, value)

from __future__ import annotations

import json
import random

import numpy as np

from mcts_chess.games.base import BaseGame
from mcts_chess.mcts.engine import MCTSEngine, MCTSNode
from mcts_chess.network.model import AlphaZeroNet


class EloRating:
    def __init__(self, k: float = 32.0) -> None:
        self.ratings: dict[str, float] = {}
        self.default_rating: float = 1000.0
        self.k: float = k
        self._history: dict[str, list[tuple[int, float]]] = {}
        self._game_counter: int = 0
        self.ratings["random_baseline"] = 0.0
        self._history["random_baseline"] = [(0, 0.0)]

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: str, loser_id: str, draw: bool = False) -> None:
        self._game_counter += 1
        rating_a = self.get_rating(winner_id)
        rating_b = self.get_rating(loser_id)
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        if draw:
            self.ratings[winner_id] = rating_a + self.k * (0.5 - expected_a)
            self.ratings[loser_id] = rating_b + self.k * (0.5 - expected_b)
        else:
            self.ratings[winner_id] = rating_a + self.k * (1.0 - expected_a)
            self.ratings[loser_id] = rating_b + self.k * (0.0 - expected_b)
        for pid in (winner_id, loser_id):
            if pid not in self._history:
                self._history[pid] = []
            self._history[pid].append((self._game_counter, self.ratings[pid]))

    def get_rating(self, player_id: str) -> float:
        return self.ratings.get(player_id, self.default_rating)

    def get_all_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    def save(self, path: str) -> None:
        data = {
            "ratings": self.ratings,
            "k": self.k,
            "default_rating": self.default_rating,
            "history": {
                pid: [(gn, r) for gn, r in entries]
                for pid, entries in self._history.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.ratings = data["ratings"]
        self.k = data.get("k", 32.0)
        self.default_rating = data.get("default_rating", 1000.0)
        self._history = {
            pid: [(gn, r) for gn, r in entries]
            for pid, entries in data.get("history", {}).items()
        }


class Arena:
    def __init__(
        self,
        game_class: type,
        num_games: int = 50,
        num_iterations: int = 400,
        c_puct: float = 1.5,
    ) -> None:
        self.game_class = game_class
        self.num_games = num_games
        self.num_iterations = num_iterations
        self.c_puct = c_puct
        self.elo = EloRating()

    def evaluate_models(
        self,
        model_challenger: AlphaZeroNet,
        model_champion: AlphaZeroNet,
        model_challenger_id: str,
        model_champion_id: str,
    ) -> dict:
        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(self.num_games):
            game = self.game_class()
            if game_idx % 2 == 0:
                result, _ = self.play_match(game, model_challenger, model_champion)
            else:
                result, _ = self.play_match(game, model_champion, model_challenger)
                result = -result
            if result > 0:
                wins += 1
                self.elo.update(model_challenger_id, model_champion_id, draw=False)
            elif result < 0:
                losses += 1
                self.elo.update(model_champion_id, model_challenger_id, draw=False)
            else:
                draws += 1
                self.elo.update(model_challenger_id, model_champion_id, draw=True)
        win_rate = wins / self.num_games if self.num_games > 0 else 0.0
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "challenger_elo": self.elo.get_rating(model_challenger_id),
            "champion_elo": self.elo.get_rating(model_champion_id),
            "accepted": win_rate > 0.55,
        }

    def evaluate_vs_random(
        self,
        model: AlphaZeroNet,
        model_id: str,
        num_games: int = 20,
    ) -> dict:
        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(num_games):
            game = self.game_class()
            if game_idx % 2 == 0:
                result, _ = self.play_match(game, model, None)
            else:
                result, _ = self.play_match(game, None, model)
                result = -result
            if result > 0:
                wins += 1
                self.elo.update(model_id, "random_baseline", draw=False)
            elif result < 0:
                losses += 1
                self.elo.update("random_baseline", model_id, draw=False)
            else:
                draws += 1
                self.elo.update(model_id, "random_baseline", draw=True)
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0.0,
        }

    def play_match(
        self,
        game: BaseGame,
        model1: AlphaZeroNet | None,
        model2: AlphaZeroNet | None,
    ) -> tuple[float, list]:
        game = game.copy()
        move_history: list[dict] = []
        while not game.is_game_over():
            current_player = game.get_current_player()
            board_before = game.board.copy()
            model = model1 if current_player == 1 else model2
            if model is not None:
                engine = MCTSEngine(
                    num_iterations=self.num_iterations,
                    c_puct=self.c_puct,
                    use_puct=True,
                    neural_net=model,
                )
                action, policy = engine.search(game)
            else:
                legal_moves = game.get_legal_moves()
                action = random.choice(legal_moves)
                policy = None
            move_history.append({
                "action": action,
                "board_before_move": board_before,
                "policy": policy,
                "player": current_player,
            })
            game = game.make_move(action)
        result = game.get_result()
        return float(result), move_history

    def get_elo_history(self) -> dict[str, list[tuple[int, float]]]:
        return {pid: list(entries) for pid, entries in self.elo._history.items()}

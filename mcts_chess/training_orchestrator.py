from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from mcts_chess import (
    AlphaZeroNet,
    Arena,
    Connect4,
    EloRating,
    GoGame,
    Gomoku,
    MCTSEngine,
    NetworkTrainer,
    Othello,
    ReplayBuffer,
    SelfPlayPipeline,
)

GAME_CONFIGS: dict[str, dict] = {
    "gomoku": {
        "class": Gomoku,
        "input_channels": 3,
        "board_size": 15,
        "action_size": 225,
        "dirichlet_alpha": 0.03,
    },
    "othello": {
        "class": Othello,
        "input_channels": 3,
        "board_size": 8,
        "action_size": 64,
        "dirichlet_alpha": 0.3 / 64,
    },
    "connect4": {
        "class": Connect4,
        "input_channels": 3,
        "board_size": 7,
        "action_size": 7,
        "dirichlet_alpha": 0.3 / 7,
        "board_height": 6,
        "board_width": 7,
    },
    "go": {
        "class": GoGame,
        "input_channels": 19,
        "board_size": 9,
        "action_size": 82,
        "dirichlet_alpha": 0.03,
    },
}


class TrainingOrchestrator:
    def __init__(
        self,
        game_type: str,
        data_dir: str = "data",
        device: str = "cpu",
        num_res_blocks: int = 5,
        channels: int = 64,
        lr: float = 0.001,
        batch_size: int = 512,
        buffer_capacity: int = 500000,
        selfplay_games: int = 100,
        selfplay_iterations: int = 400,
        selfplay_parallel: int = 4,
        eval_games: int = 50,
        eval_iterations: int = 400,
        c_puct: float = 1.5,
    ) -> None:
        self.game_type = game_type
        config = GAME_CONFIGS[game_type]
        self.game_class: type = config["class"]

        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        model_kwargs = {
            "input_channels": config["input_channels"],
            "board_size": config["board_size"],
            "action_size": config["action_size"],
            "num_res_blocks": num_res_blocks,
            "channels": channels,
        }
        if "board_height" in config:
            model_kwargs["board_height"] = config["board_height"]
        if "board_width" in config:
            model_kwargs["board_width"] = config["board_width"]
        self.model = AlphaZeroNet(**model_kwargs).to(device)

        self.trainer = NetworkTrainer(self.model, lr=lr, device=device)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.pipeline = SelfPlayPipeline(
            game_class=self.game_class,
            model=self.model,
            buffer=self.buffer,
            num_parallel=selfplay_parallel,
            num_iterations=selfplay_iterations,
            c_puct=c_puct,
            device=device,
        )
        self.arena = Arena(
            game_class=self.game_class,
            num_games=eval_games,
            num_iterations=eval_iterations,
            c_puct=c_puct,
        )

        self.batch_size = batch_size
        self.selfplay_games = selfplay_games
        self.device = device
        self.c_puct = c_puct

        best_model_path = self.models_dir / "best_model.pt"
        if best_model_path.exists():
            self.trainer.load_checkpoint(str(best_model_path))

        self.version: int = 0
        self.training_log: list[dict] = self._load_training_log()

    def train(self, num_iterations: int = 10) -> None:
        for iteration in range(num_iterations):
            iter_start = time.time()

            self.pipeline.generate(num_games=self.selfplay_games)

            num_steps = min(len(self.buffer) // self.batch_size, 100)
            iteration_losses: list[dict] = []
            for _ in range(num_steps):
                states, policies, values = self.buffer.sample(self.batch_size)
                losses = self.trainer.train_step(states, policies, values)
                iteration_losses.append(losses)

            checkpoint_path = self.models_dir / f"v{self.version}.pt"
            self.trainer.save_checkpoint(str(checkpoint_path))

            accepted = True
            eval_result: dict | None = None
            best_model_path = self.models_dir / "best_model.pt"
            if self.version == 0:
                self.trainer.save_checkpoint(str(best_model_path))
            else:
                best_model_path = self.models_dir / "best_model.pt"
                best_model_kwargs = {
                    "input_channels": self.model.conv_in.in_channels,
                    "board_size": self.model.board_size,
                    "action_size": self.model.action_size,
                    "num_res_blocks": len(self.model.res_blocks),
                    "channels": self.model.conv_in.out_channels,
                }
                if hasattr(self.model, "board_height") and self.model.board_height != self.model.board_size:
                    best_model_kwargs["board_height"] = self.model.board_height
                    best_model_kwargs["board_width"] = self.model.board_width
                best_model = AlphaZeroNet(**best_model_kwargs).to(self.device)
                best_checkpoint = torch.load(
                    str(best_model_path), map_location=self.device, weights_only=True
                )
                best_model.load_state_dict(best_checkpoint["model_state_dict"])
                best_model.eval()

                challenger_id = f"v{self.version}"
                champion_id = "best"
                eval_result = self.arena.evaluate_models(
                    model_challenger=self.model,
                    model_champion=best_model,
                    model_challenger_id=challenger_id,
                    model_champion_id=champion_id,
                )
                accepted = eval_result["accepted"]

                if accepted:
                    self.trainer.save_checkpoint(str(best_model_path))
                else:
                    prev_checkpoint_path = self.models_dir / "best_model.pt"
                    self.trainer.load_checkpoint(str(prev_checkpoint_path))

            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_total_loss = 0.0
            if iteration_losses:
                avg_policy_loss = np.mean([l["policy_loss"] for l in iteration_losses])
                avg_value_loss = np.mean([l["value_loss"] for l in iteration_losses])
                avg_total_loss = np.mean([l["total_loss"] for l in iteration_losses])

            log_entry: dict = {
                "version": self.version,
                "iteration": iteration,
                "buffer_size": len(self.buffer),
                "num_train_steps": num_steps,
                "avg_policy_loss": float(avg_policy_loss),
                "avg_value_loss": float(avg_value_loss),
                "avg_total_loss": float(avg_total_loss),
                "accepted": accepted,
                "elapsed_seconds": time.time() - iter_start,
            }
            if eval_result is not None:
                log_entry["eval_win_rate"] = eval_result["win_rate"]
                log_entry["eval_wins"] = eval_result["wins"]
                log_entry["eval_losses"] = eval_result["losses"]
                log_entry["eval_draws"] = eval_result["draws"]
                log_entry["challenger_elo"] = eval_result["challenger_elo"]
                log_entry["champion_elo"] = eval_result["champion_elo"]

            self.training_log.append(log_entry)
            self._save_training_log()

            elo_data = {
                pid: rating
                for pid, rating in self.arena.elo.ratings.items()
            }
            elo_path = self.data_dir / "elo_ratings.json"
            with open(elo_path, "w") as f:
                json.dump(elo_data, f, indent=2)

            self.version += 1

    def _save_training_log(self) -> None:
        log_path = self.data_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def _load_training_log(self) -> list[dict]:
        log_path = self.data_dir / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                return json.load(f)
        return []

    def _save_game_record(self, record: dict) -> None:
        records_path = self.data_dir / "game_records.json"
        records: list[dict] = []
        if records_path.exists():
            with open(records_path) as f:
                records = json.load(f)
        records.append(record)
        with open(records_path, "w") as f:
            json.dump(records, f, indent=2)

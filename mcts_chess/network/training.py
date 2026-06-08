from __future__ import annotations

import numpy as np
import torch
from mcts_chess.network.model import AlphaZeroNet, AlphaZeroLoss


class NetworkTrainer:
    def __init__(
        self,
        model: AlphaZeroNet,
        lr: float = 0.001,
        l2_reg: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.l2_reg = l2_reg
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = AlphaZeroLoss()

    def train_step(
        self,
        states: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> dict:
        self.model.train()
        states_t = torch.from_numpy(states).float().to(self.device)
        policy_targets_t = torch.from_numpy(policy_targets).float().to(self.device)
        value_targets_t = torch.from_numpy(value_targets).float().to(self.device).unsqueeze(1)

        policy_logits, value_pred = self.model(states_t)

        policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets_t)
        value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(-1), value_targets_t.squeeze(-1))

        loss = self.loss_fn(
            policy_logits, policy_targets_t, value_pred, value_targets_t, l2_reg=self.l2_reg
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(loss.item()),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

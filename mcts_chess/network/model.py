import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        board_size: int,
        action_size: int,
        num_res_blocks: int = 5,
        channels: int = 64,
        board_height: int | None = None,
        board_width: int | None = None,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.board_height = board_height if board_height is not None else board_size
        self.board_width = board_width if board_width is not None else board_size
        self.action_size = action_size

        self.conv_in = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu_in = nn.ReLU()
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        policy_flat_size = 2 * self.board_height * self.board_width
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(policy_flat_size, action_size)

        value_flat_size = 1 * self.board_height * self.board_width
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU()
        self.value_fc1 = nn.Linear(value_flat_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.relu_in(self.bn_in(self.conv_in(x)))
        out = self.res_blocks(out)

        policy = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = self.value_relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = self.value_tanh(self.value_fc2(F.relu(self.value_fc1(value))))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        if board.ndim == 2:
            tensor = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0)
        elif board.ndim == 3:
            tensor = torch.from_numpy(board).float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(board).float()
        device = next(self.parameters()).device
        tensor = tensor.to(device)
        self.eval()
        policy_logits, value = self.forward(tensor)
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value_float = value.cpu().numpy()[0, 0]
        return policy_probs, float(value_float)


class AlphaZeroLoss(nn.Module):
    def forward(
        self,
        policy_logits: torch.Tensor,
        policy_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> torch.Tensor:
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_pred.view(-1), value_target.view(-1))
        l2_loss = sum(p.norm(2).pow(2) for p in self.parameters())
        return policy_loss + value_loss + l2_reg * l2_loss

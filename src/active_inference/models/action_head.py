import torch
import torch.nn as nn
from torch import Tensor


class ActionHead(nn.Module):
    def __init__(self, feat_dim: int = 320, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self._net(feat)

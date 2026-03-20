import torch
import torch.nn as nn
from torch import Tensor


class EnsembleTransitionHeads(nn.Module):
    def __init__(
        self, feat_dim: int = 320, stoch_dim: int = 64, hidden_dim: int = 256, num_heads: int = 5
    ):
        super().__init__()
        self._heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, stoch_dim),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, feat: Tensor) -> Tensor:
        return torch.stack([head(feat) for head in self._heads])

    def epistemic_uncertainty(self, feat: Tensor) -> Tensor:
        preds = self.forward(feat)
        return preds.var(dim=0).mean(dim=-1)

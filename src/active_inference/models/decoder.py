import torch.nn as nn
from torch import Tensor


class ObsDecoder(nn.Module):
    def __init__(self, feat_dim: int = 320, image_channels: int = 3):
        super().__init__()
        self._fc = nn.Sequential(nn.Linear(feat_dim, 256 * 4 * 4), nn.ReLU())
        self._deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),
        )

    def forward(self, feat: Tensor) -> Tensor:
        x = self._fc(feat)
        x = x.reshape(x.shape[0], 256, 4, 4)
        return self._deconv(x)


class StateDecoder(nn.Module):
    def __init__(self, feat_dim: int = 320, state_dim: int = 2):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self._mlp(feat)

import torch
import torch.nn as nn
from torch import Tensor

from active_inference.utils.transforms import crop_road as _crop_road


class ConvEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        state_dim: int = 2,
        embed_dim: int = 256,
        crop_road: bool = False,
    ):
        super().__init__()
        self._crop_road = crop_road
        self._cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self._img_proj = nn.Sequential(nn.Linear(256 * 4 * 4, embed_dim), nn.ReLU())
        self._state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self._fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())

    def forward(self, img: Tensor, state: Tensor) -> Tensor:
        if self._crop_road:
            img = _crop_road(img)
        x = self._cnn(img)
        x = x.reshape(x.shape[0], -1)
        img_embed = self._img_proj(x)
        state_embed = self._state_mlp(state)
        return self._fusion(torch.cat([img_embed, state_embed], dim=-1))

import torch
import torch.nn.functional as F
from torch import Tensor


def crop_road(img: Tensor, keep_bottom_frac: float = 0.6) -> Tensor:
    """Crop top portion (sky/buildings), keep bottom for road.

    Input:  [C, H, W] or [B, C, H, W]
    Output: same shape (resized back to original H, W)
    """
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)
    _, _, h, w = img.shape
    start_row = int(h * (1.0 - keep_bottom_frac))
    cropped = img[:, :, start_row:, :]
    resized = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
    return resized.squeeze(0) if squeeze else resized


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def normalize_image(x: Tensor) -> Tensor:
    return x.float() / 255.0 - 0.5


def denormalize_image(x: Tensor) -> Tensor:
    return ((x + 0.5) * 255.0).clamp(0, 255).to(torch.uint8)

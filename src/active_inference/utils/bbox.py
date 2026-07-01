from __future__ import annotations

import torch
from torch import Tensor


def _as_bbox_tensor(bbox) -> tuple[Tensor, torch.Size]:
    out = torch.as_tensor(bbox, dtype=torch.float32)
    if out.shape[-1] != 4:
        raise ValueError(f"Expected bbox last dim 4, got shape={tuple(out.shape)}")
    return out, out.shape


def bbox_xyxy_raw_to_crop_road(
    bbox,
    image_h: int,
    image_w: int,
    keep_bottom_frac: float = 0.6,
    min_area: float = 1.0,
) -> Tensor:
    """Map raw/original xyxy boxes into crop_road model-image coordinates.

    Supports [..., 4] Tensor-like input. Invalid boxes are returned as NaNs.
    """
    b, orig_shape = _as_bbox_tensor(bbox)
    flat = b.reshape(-1, 4)
    out = torch.full_like(flat, float("nan"))

    start_row = int(image_h * (1.0 - keep_bottom_frac))
    kept_h = image_h - start_row
    if kept_h <= 0:
        return out.reshape(orig_shape)

    finite = torch.isfinite(flat).all(dim=-1)
    x1, y1, x2, y2 = flat.unbind(dim=-1)
    valid = finite & (x2 > x1) & (y2 > y1) & (y2 > start_row)

    x1p = x1.clamp(0.0, float(image_w))
    x2p = x2.clamp(0.0, float(image_w))
    y1_crop = torch.maximum(y1, torch.tensor(float(start_row), device=flat.device))
    y2_crop = torch.minimum(y2, torch.tensor(float(image_h), device=flat.device))
    y1p = (y1_crop - start_row) * image_h / kept_h
    y2p = (y2_crop - start_row) * image_h / kept_h
    y1p = y1p.clamp(0.0, float(image_h))
    y2p = y2p.clamp(0.0, float(image_h))

    area = (x2p - x1p) * (y2p - y1p)
    valid = valid & (x2p > x1p) & (y2p > y1p) & (area >= min_area)
    mapped = torch.stack([x1p, y1p, x2p, y2p], dim=-1)
    out[valid] = mapped[valid]
    return out.reshape(orig_shape)


def bbox_xyxy_to_model_space(
    bbox,
    image_h: int,
    image_w: int,
    crop_road: bool = False,
    keep_bottom_frac: float = 0.6,
    coord_space: str = "raw",
    min_area: float = 1.0,
) -> Tensor:
    """Map dataset bbox coords into model/preprocessed image space."""
    if coord_space == "model" or not crop_road:
        b, orig_shape = _as_bbox_tensor(bbox)
        flat = b.reshape(-1, 4)
        out = torch.full_like(flat, float("nan"))
        finite = torch.isfinite(flat).all(dim=-1)
        x1, y1, x2, y2 = flat.unbind(dim=-1)
        x1 = x1.clamp(0.0, float(image_w))
        x2 = x2.clamp(0.0, float(image_w))
        y1 = y1.clamp(0.0, float(image_h))
        y2 = y2.clamp(0.0, float(image_h))
        area = (x2 - x1) * (y2 - y1)
        valid = finite & (x2 > x1) & (y2 > y1) & (area >= min_area)
        clipped = torch.stack([x1, y1, x2, y2], dim=-1)
        out[valid] = clipped[valid]
        return out.reshape(orig_shape)
    if coord_space != "raw":
        raise ValueError(f"coord_space must be 'raw' or 'model', got {coord_space!r}")
    return bbox_xyxy_raw_to_crop_road(
        bbox,
        image_h=image_h,
        image_w=image_w,
        keep_bottom_frac=keep_bottom_frac,
        min_area=min_area,
    )


def bbox_xyxy_to_mask(
    bbox,
    image_h: int,
    image_w: int,
    dilate: int = 0,
    min_area: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Convert model-space xyxy boxes to masks and validity flags.

    Returns:
        mask:  [N, 1, image_h, image_w], where N is product of leading dims.
        valid: [N] bool.
    """
    b, _ = _as_bbox_tensor(bbox)
    flat = b.reshape(-1, 4)
    n = flat.shape[0]
    mask = torch.zeros(n, 1, image_h, image_w, device=flat.device)
    valid_out = torch.zeros(n, device=flat.device, dtype=torch.bool)
    if n == 0:
        return mask, valid_out

    finite = torch.isfinite(flat).all(dim=-1)
    x1, y1, x2, y2 = flat.unbind(dim=-1)
    d = max(0, int(dilate))
    x1 = (x1 - d).clamp(0.0, float(image_w))
    x2 = (x2 + d).clamp(0.0, float(image_w))
    y1 = (y1 - d).clamp(0.0, float(image_h))
    y2 = (y2 + d).clamp(0.0, float(image_h))
    area = (x2 - x1) * (y2 - y1)
    valid = finite & (x2 > x1) & (y2 > y1) & (area >= min_area)

    ys = torch.arange(image_h, device=flat.device).view(1, image_h, 1)
    xs = torch.arange(image_w, device=flat.device).view(1, 1, image_w)
    for i in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        inside = (xs >= x1[i]) & (xs < x2[i]) & (ys >= y1[i]) & (ys < y2[i])
        if inside.any():
            mask[i, 0] = inside.float()
            valid_out[i] = True
    return mask, valid_out


def bbox_xyxy_to_token_mask(
    bbox,
    image_h: int,
    image_w: int,
    patch_size: int = 8,
    min_area: float = 1.0,
) -> Tensor:
    """Convert model-space xyxy boxes to patch-token overlap masks."""
    pixel_mask, valid = bbox_xyxy_to_mask(
        bbox, image_h, image_w, dilate=0, min_area=min_area,
    )
    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    token_mask = pixel_mask.reshape(
        pixel_mask.shape[0],
        1,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    ).amax(dim=(1, 3, 5)).reshape(pixel_mask.shape[0], grid_h * grid_w)
    return token_mask.bool() & valid.unsqueeze(-1)

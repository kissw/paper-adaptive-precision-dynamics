#!/usr/bin/env python3
"""Visualize obstacle_bbox coordinate conventions.

This script draws the stored bbox in multiple coordinate assumptions:

1. Raw image + stored bbox as-is
2. Model/preprocessed image + stored bbox as-is
3. Model/preprocessed image + bbox transformed from raw-space to crop_road-space
4. Model/preprocessed image + both boxes + ViT patch grid/token overlap

Use this to decide whether the HDF5 obstacle_bbox is already in model image
space or still in raw image space.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    from active_inference.config import Config
    from active_inference.utils.bbox import bbox_xyxy_to_model_space
    from active_inference.utils.transforms import crop_road as crop_road_fn
except Exception:
    Config = None
    bbox_xyxy_to_model_space = None
    crop_road_fn = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="HDF5 dataset path")
    p.add_argument("--config", default=None, help="Optional yaml config")
    p.add_argument("--output_dir", default="outputs/debug_bbox_overlay")
    p.add_argument("--num", type=int, default=16, help="Number of frames to save")
    p.add_argument("--start_idx", type=int, default=None, help="Start frame index")
    p.add_argument("--indices", default=None, help="Comma-separated explicit frame indices")
    p.add_argument("--visible_only", action="store_true", help="Use only obstacle_visible==1 frames")
    p.add_argument("--crop_road", action="store_true", help="Force crop_road visualization")
    p.add_argument("--no_crop_road", action="store_true", help="Disable crop_road visualization")
    p.add_argument("--keep_bottom_frac", type=float, default=0.6)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def infer_crop_road(args) -> bool:
    if args.crop_road and args.no_crop_road:
        raise ValueError("Use only one of --crop_road or --no_crop_road")

    if args.crop_road:
        return True
    if args.no_crop_road:
        return False

    if args.config and Config is not None:
        cfg = Config.from_yaml(args.config)
        return bool(getattr(cfg.encoder, "crop_road", False))

    return False


def infer_bbox_coord_space(args) -> str:
    if args.config and Config is not None:
        cfg = Config.from_yaml(args.config)
        return getattr(cfg.training, "bbox_coord_space", "raw")
    return "raw"


def to_chw(img: np.ndarray) -> np.ndarray:
    """Return CHW image."""
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={img.shape}")
    if img.shape[0] in (1, 3):
        return img
    if img.shape[-1] in (1, 3):
        return np.transpose(img, (2, 0, 1))
    raise ValueError(f"Cannot infer image layout from shape={img.shape}")


def to_display(img_chw: np.ndarray) -> np.ndarray:
    """Convert CHW image in uint8/[0,1]/[-0.5,0.5] to HWC [0,1]."""
    x = img_chw.astype(np.float32)

    if x.max() > 2.0:
        x = x / 255.0
    elif x.min() < -0.05:
        x = x + 0.5

    x = np.clip(x, 0.0, 1.0)
    return np.transpose(x, (1, 2, 0))


def apply_crop_road_np(img_chw: np.ndarray, keep_bottom_frac: float) -> np.ndarray:
    """Apply the repo crop_road transform to a CHW numpy image."""
    if crop_road_fn is not None:
        t = torch.from_numpy(img_chw).float()
        out = crop_road_fn(t, keep_bottom_frac=keep_bottom_frac)
        return out.detach().cpu().numpy()

    # Fallback implementation, nearest-ish via torch interpolate.
    t = torch.from_numpy(img_chw).float().unsqueeze(0)
    _, _, h, w = t.shape
    start_row = int(h * (1.0 - keep_bottom_frac))
    cropped = t[:, :, start_row:, :]
    out = torch.nn.functional.interpolate(
        cropped, size=(h, w), mode="bilinear", align_corners=False
    )
    return out.squeeze(0).numpy()


def valid_bbox(b: np.ndarray) -> bool:
    if b is None:
        return False
    if b.shape[0] != 4:
        return False
    if not np.isfinite(b).all():
        return False
    x1, y1, x2, y2 = b.astype(float)
    return (x2 > x1) and (y2 > y1)


def clip_bbox(b: np.ndarray, w: int, h: int) -> np.ndarray | None:
    if not valid_bbox(b):
        return None
    x1, y1, x2, y2 = b.astype(float)
    x1 = float(np.clip(x1, 0, w))
    x2 = float(np.clip(x2, 0, w))
    y1 = float(np.clip(y1, 0, h))
    y2 = float(np.clip(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def raw_bbox_to_crop_model_bbox(
    b: np.ndarray,
    w: int,
    h: int,
    keep_bottom_frac: float = 0.6,
) -> np.ndarray | None:
    """Map bbox from raw image space to crop_road output image space.

    crop_road keeps rows [start_row:h] and resizes them back to H.
    x is unchanged except clipping.
    y is shifted by start_row and scaled by h / (h - start_row).
    """
    if not valid_bbox(b):
        return None

    start_row = int(h * (1.0 - keep_bottom_frac))
    kept_h = h - start_row
    if kept_h <= 0:
        return None

    x1, y1, x2, y2 = b.astype(float)

    # Intersect with kept vertical crop first.
    y1 = max(y1, start_row)
    y2 = min(y2, h)
    if y2 <= y1:
        return None

    scale_y = h / kept_h
    y1p = (y1 - start_row) * scale_y
    y2p = (y2 - start_row) * scale_y

    out = np.array([x1, y1p, x2, y2p], dtype=np.float32)
    return clip_bbox(out, w, h)


def training_space_bbox(
    b: np.ndarray,
    w: int,
    h: int,
    crop_road: bool,
    keep_bottom_frac: float,
    coord_space: str,
) -> np.ndarray | None:
    if bbox_xyxy_to_model_space is not None:
        out = bbox_xyxy_to_model_space(
            torch.as_tensor(b, dtype=torch.float32),
            image_h=h,
            image_w=w,
            crop_road=crop_road,
            keep_bottom_frac=keep_bottom_frac,
            coord_space=coord_space,
            min_area=1.0,
        ).detach().cpu().numpy()
        return clip_bbox(out, w, h)
    if coord_space == "model" or not crop_road:
        return clip_bbox(b, w, h)
    if coord_space != "raw":
        raise ValueError(f"Unknown bbox coord space: {coord_space!r}")
    return raw_bbox_to_crop_model_bbox(
        b, w=w, h=h, keep_bottom_frac=keep_bottom_frac,
    )


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else float("nan")


def token_indices_for_bbox(
    b: np.ndarray | None,
    w: int,
    h: int,
    patch_size: int,
) -> list[int]:
    if b is None:
        return []
    x1, y1, x2, y2 = b
    grid_w = w // patch_size
    grid_h = h // patch_size

    c1 = int(math.floor(x1 / patch_size))
    c2 = int(math.ceil(x2 / patch_size)) - 1
    r1 = int(math.floor(y1 / patch_size))
    r2 = int(math.ceil(y2 / patch_size)) - 1

    c1, c2 = max(0, c1), min(grid_w - 1, c2)
    r1, r2 = max(0, r1), min(grid_h - 1, r2)

    if c2 < c1 or r2 < r1:
        return []

    return [r * grid_w + c for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]


def draw_bbox(ax, b, color, label, lw=2):
    import matplotlib.patches as patches

    if b is None:
        return
    x1, y1, x2, y2 = b
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        label=label,
    )
    ax.add_patch(rect)


def draw_patch_grid(ax, w: int, h: int, patch_size: int):
    for x in range(0, w + 1, patch_size):
        ax.axvline(x - 0.5, color="white", alpha=0.25, linewidth=0.5)
    for y in range(0, h + 1, patch_size):
        ax.axhline(y - 0.5, color="white", alpha=0.25, linewidth=0.5)


def draw_token_highlights(ax, token_ids: list[int], w: int, h: int, patch_size: int, color: str):
    import matplotlib.patches as patches

    grid_w = w // patch_size
    for tid in token_ids:
        r = tid // grid_w
        c = tid % grid_w
        rect = patches.Rectangle(
            (c * patch_size, r * patch_size),
            patch_size,
            patch_size,
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(rect)


def choose_indices(f: h5py.File, args) -> list[int]:
    n = len(f["images"])

    if args.indices:
        idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        return [i for i in idxs if 0 <= i < n]

    if args.visible_only and "obstacle_visible" in f:
        visible = f["obstacle_visible"][:].astype(bool)
        idxs = np.where(visible)[0].tolist()
    else:
        idxs = list(range(n))

    # Also prefer finite bbox if available.
    if "obstacle_bbox" in f:
        bboxes = f["obstacle_bbox"][:]
        finite = np.isfinite(bboxes).all(axis=1)
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        valid = finite & (area > 1.0)
        idxs = [i for i in idxs if valid[i]]

    if args.start_idx is not None:
        idxs = [i for i in idxs if i >= args.start_idx]

    if not idxs:
        return []

    if len(idxs) <= args.num:
        return idxs

    # Spread samples across available visible frames.
    pos = np.linspace(0, len(idxs) - 1, args.num).round().astype(int)
    return [idxs[p] for p in pos]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_crop = infer_crop_road(args)
    bbox_coord_space = infer_bbox_coord_space(args)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(args.data, "r") as f:
        if "obstacle_bbox" not in f:
            raise KeyError("HDF5 has no obstacle_bbox dataset")

        idxs = choose_indices(f, args)
        if not idxs:
            raise RuntimeError("No valid bbox frames found")

        print(f"data={args.data}")
        print(f"output_dir={out_dir}")
        print(f"crop_road={do_crop}, keep_bottom_frac={args.keep_bottom_frac}")
        print(f"bbox_coord_space={bbox_coord_space}")
        print(f"indices={idxs}")

        summary_lines = [
            "idx,visible,raw_bbox,direct_model_bbox,raw_to_crop_bbox,training_space_bbox,iou_direct_vs_training,tokens_direct,tokens_training"
        ]

        for idx in idxs:
            img = to_chw(f["images"][idx])
            bbox_stored = f["obstacle_bbox"][idx].astype(np.float32)

            _, h, w = img.shape
            raw_disp = to_display(img)

            if do_crop:
                model_img = apply_crop_road_np(img, args.keep_bottom_frac)
            else:
                model_img = img.copy()
            model_disp = to_display(model_img)

            b_raw_direct = clip_bbox(bbox_stored, w, h)
            b_model_direct = clip_bbox(bbox_stored, w, h)

            if do_crop:
                b_raw_to_model = raw_bbox_to_crop_model_bbox(
                    bbox_stored, w=w, h=h, keep_bottom_frac=args.keep_bottom_frac
                )
            else:
                b_raw_to_model = clip_bbox(bbox_stored, w, h)
            b_training = training_space_bbox(
                bbox_stored,
                w=w,
                h=h,
                crop_road=do_crop,
                keep_bottom_frac=args.keep_bottom_frac,
                coord_space=bbox_coord_space,
            )

            tokens_direct = token_indices_for_bbox(b_model_direct, w, h, args.patch_size)
            tokens_training = token_indices_for_bbox(b_training, w, h, args.patch_size)
            iou = bbox_iou(b_model_direct, b_training)

            visible = None
            if "obstacle_visible" in f:
                visible = int(f["obstacle_visible"][idx])

            fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=args.dpi)
            axes = axes.ravel()

            axes[0].imshow(raw_disp)
            draw_bbox(axes[0], b_raw_direct, "lime", "stored bbox")
            axes[0].set_title(f"raw image + stored bbox\nidx={idx}, visible={visible}")
            axes[0].axis("off")

            axes[1].imshow(model_disp)
            draw_bbox(axes[1], b_model_direct, "lime", "direct/model-space")
            axes[1].set_title("preprocessed image + stored bbox as-is\nGREEN = assume model-space")
            axes[1].axis("off")

            axes[2].imshow(model_disp)
            draw_bbox(axes[2], b_training, "red", "training-space bbox")
            axes[2].set_title(
                "preprocessed image + training-space bbox\n"
                f"RED = coord_space={bbox_coord_space}"
            )
            axes[2].axis("off")

            axes[3].imshow(model_disp)
            draw_patch_grid(axes[3], w, h, args.patch_size)
            draw_token_highlights(axes[3], tokens_direct, w, h, args.patch_size, "lime")
            draw_token_highlights(axes[3], tokens_training, w, h, args.patch_size, "red")
            draw_bbox(axes[3], b_model_direct, "lime", "direct/model-space", lw=2)
            draw_bbox(axes[3], b_training, "red", "training-space", lw=2)
            axes[3].set_title(
                f"both + {args.patch_size}x{args.patch_size} token grid\n"
                f"green tokens={len(tokens_direct)}, red tokens={len(tokens_training)}, IoU={iou:.3f}"
            )
            axes[3].axis("off")

            for ax in axes:
                ax.set_xlim(-0.5, w - 0.5)
                ax.set_ylim(h - 0.5, -0.5)

            fig.tight_layout()
            out_path = out_dir / f"bbox_overlay_idx{idx:06d}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

            summary_lines.append(
                f"{idx},{visible},{bbox_stored.tolist()},"
                f"{None if b_model_direct is None else b_model_direct.tolist()},"
                f"{None if b_raw_to_model is None else b_raw_to_model.tolist()},"
                f"{None if b_training is None else b_training.tolist()},"
                f"{iou},"
                f"{tokens_direct},"
                f"{tokens_training}"
            )
            print(f"saved {out_path}")

    summary_path = out_dir / "bbox_overlay_summary.csv"
    summary_path.write_text("\n".join(summary_lines))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()

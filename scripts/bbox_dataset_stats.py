import argparse

import h5py
import torch

from active_inference.config import Config
from active_inference.utils.bbox import (
    bbox_xyxy_to_model_space,
    bbox_xyxy_to_token_mask,
)


def _target_rollout_horizons(cfg: Config) -> list[int]:
    horizon = int(getattr(cfg.training, "rollout_horizon", 1))
    decode = getattr(cfg.training, "rollout_decode_horizons", None)
    if decode:
        horizons = [int(h) for h in decode]
    else:
        horizons = [1, horizon]
    horizons = [h for h in horizons if 1 <= h <= horizon]
    return horizons or [1]


def _model_bboxes(raw_bbox: torch.Tensor, cfg: Config) -> torch.Tensor:
    image_size = int(getattr(cfg.encoder, "image_size", 64))
    return bbox_xyxy_to_model_space(
        raw_bbox,
        image_h=image_size,
        image_w=image_size,
        crop_road=bool(getattr(cfg.encoder, "crop_road", False)),
        keep_bottom_frac=float(getattr(cfg.encoder, "keep_bottom_frac", 0.6)),
        coord_space=getattr(cfg.training, "bbox_coord_space", "raw"),
        min_area=float(getattr(cfg.training, "bbox_min_area", 1.0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage", default="transition", choices=["joint", "ae", "transition"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    cfg.training.stage = args.stage
    if args.device:
        cfg.device = args.device

    with h5py.File(args.data, "r") as f:
        total = len(f["episode_ids"])
        episode_ids = f["episode_ids"][:]
        obstacle_visible = (
            torch.from_numpy(f["obstacle_visible"][:].astype("float32"))
            if "obstacle_visible" in f else None
        )
        raw_bbox = (
            torch.from_numpy(f["obstacle_bbox"][:].astype("float32"))
            if "obstacle_bbox" in f else torch.full((total, 4), float("nan"))
        )

    model_bbox = _model_bboxes(raw_bbox, cfg)
    valid = torch.isfinite(model_bbox).all(dim=-1)
    area = (
        (model_bbox[:, 2] - model_bbox[:, 0])
        * (model_bbox[:, 3] - model_bbox[:, 1])
    )
    valid_area = area[valid]

    image_size = int(getattr(cfg.encoder, "image_size", 64))
    patch_size = int(getattr(cfg.token_vit, "patch_size", 8))
    token_mask = bbox_xyxy_to_token_mask(
        model_bbox,
        image_h=image_size,
        image_w=image_size,
        patch_size=patch_size,
        min_area=float(getattr(cfg.training, "bbox_min_area", 1.0)),
    )
    token_counts = token_mask.sum(dim=-1).float()

    context = int(getattr(cfg.training, "rollout_context_frames", 25)) - 1
    horizons = _target_rollout_horizons(cfg)
    context_indices = []
    for idx in range(total):
        if idx < context:
            continue
        ep = episode_ids[idx]
        if all(idx + h < total and episode_ids[idx + h] == ep for h in horizons):
            context_indices.append(idx)

    per_h = {}
    any_valid = 0
    all_valid = 0
    for idx in context_indices:
        vals = [bool(valid[idx + h]) for h in horizons]
        any_valid += int(any(vals))
        all_valid += int(all(vals))
        for h, v in zip(horizons, vals):
            per_h[h] = per_h.get(h, 0) + int(v)

    denom = max(1, len(context_indices))
    print(f"data: {args.data}")
    print(f"total_frames: {total}")
    if obstacle_visible is None:
        print("obstacle_visible_frame_ratio: nan (missing obstacle_visible)")
    else:
        print(f"obstacle_visible_frame_ratio: {float(obstacle_visible.mean()):.6f}")
    print(f"valid_bbox_ratio: {float(valid.float().mean()):.6f}")
    print(f"avg_bbox_area: {float(valid_area.mean()) if valid_area.numel() else 0.0:.6f}")
    print(
        "avg_bbox_token_count_after_crop: "
        f"{float(token_counts[valid].mean()) if valid.any() else 0.0:.6f}"
    )
    print(f"target_rollout_context_frames: {getattr(cfg.training, 'rollout_context_frames', 25)}")
    print(f"target_rollout_horizons: {horizons}")
    print(f"context_positions: {len(context_indices)}")
    print(f"valid_bbox_ratio_context_any_h: {any_valid / denom:.6f}")
    print(f"valid_bbox_ratio_context_all_h: {all_valid / denom:.6f}")
    for h in horizons:
        print(f"valid_bbox_ratio_context_plus_{h}: {per_h.get(h, 0) / denom:.6f}")


if __name__ == "__main__":
    main()

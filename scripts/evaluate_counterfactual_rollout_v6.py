#!/usr/bin/env python3
"""Evaluate counterfactual v6 open-loop rollout and save GT-vs-pred grids.

This script visualizes whether a trained RSSM / TokenViT world model reacts to
counterfactual steering branches.

Protocol
--------
For each selected cf_anchor_id and cf_branch_id:

1. Read one 50-frame counterfactual sequence.
   - context frames: cf_branch_step <= 0
   - branch frames : cf_branch_step >= 1

2. Posterior-update the model through the context frames using obs_step().
   This builds the belief at the anchor frame.

3. Open-loop rollout from the anchor belief using branch actions only.
   This uses img_step() without future observations.

4. Decode predicted future images and save a grid:
   - one GT row and one Pred row for each branch
   - columns show anchor and selected future steps.

Important convention
--------------------
The v5/v6 collector convention is:

    actions[t] is the action applied by env.step(actions[t]) that produced
    images[t], states[t].

Therefore:
    - context obs_step uses actions at the same context timestep.
    - branch img_step with action at branch step h predicts image at branch step h.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def parse_int_list(text: str | None) -> list[int] | None:
    """Parse comma-separated integer list."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def get_font(size: int) -> ImageFont.ImageFont:
    """Return a usable font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def to_uint8_hwc(img: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert CHW/HWC float image to uint8 HWC."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().float().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={arr.shape}")

    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)

    arr = arr.astype(np.float32)

    if arr.max() <= 1.0 and arr.min() >= -0.6:
        if arr.min() < 0.0:
            arr = arr + 0.5
        arr = arr * 255.0

    return np.clip(arr, 0, 255).astype(np.uint8)


def resize_tile(img: np.ndarray | torch.Tensor, tile_size: int) -> Image.Image:
    """Convert image to PIL tile."""
    pil = Image.fromarray(to_uint8_hwc(img))
    if pil.size != (tile_size, tile_size):
        pil = pil.resize((tile_size, tile_size), Image.BILINEAR)
    return pil


def deterministic_state(state):
    """Replace stochastic sample with mean for deterministic rollout.

    RSSMState:
        deter, stoch, mean, std

    TokenRSSMState:
        deter, stoch, mean, std, token_mean, token_std
    """
    if hasattr(state, "_fields") and "token_mean" in state._fields:
        return type(state)(
            state.deter,
            state.token_mean,
            state.mean,
            state.std,
            state.token_mean,
            state.token_std,
        )
    return type(state)(
        state.deter,
        state.mean,
        state.mean,
        state.std,
    )


def safe_get_frame_value(arrays: dict[str, np.ndarray], key: str, idx: int, default: Any) -> Any:
    if key not in arrays:
        return default
    return arrays[key][idx]


# ---------------------------------------------------------------------------
# HDF5 selection
# ---------------------------------------------------------------------------

def load_metadata(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load all non-image datasets into memory."""
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key == "images":
                continue
            if isinstance(f[key], h5py.Dataset):
                arrays[key] = f[key][:]
        attrs = dict(f.attrs)

    required = [
        "states",
        "actions",
        "episode_ids",
        "task_labels",
        "cf_anchor_id",
        "cf_branch_id",
        "cf_branch_step",
        "cf_bin_id",
        "cf_sampled_steer",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    return arrays, attrs


def select_anchor_ids(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    anchor_ids: list[int] | None,
    max_anchors: int | None,
) -> list[int]:
    """Select anchors filtered by task label."""
    all_anchor_ids = np.unique(arrays["cf_anchor_id"].astype(np.int64))
    selected: list[int] = []

    for anchor_id in all_anchor_ids:
        mask = arrays["cf_anchor_id"] == anchor_id
        labels = arrays["task_labels"][mask]
        task_label = int(np.round(np.mean(labels))) if labels.size else 0

        if mode == "clean" and task_label != 0:
            continue
        if mode == "obstacle" and task_label != 1:
            continue

        selected.append(int(anchor_id))

    if anchor_ids is not None:
        allowed = set(anchor_ids)
        selected = [a for a in selected if a in allowed]

    selected = sorted(selected)
    if max_anchors is not None:
        selected = selected[:max_anchors]

    if not selected:
        raise RuntimeError("No anchors selected. Check --mode, --anchor_ids, or --max_anchors.")

    return selected


def branch_sort_key(arrays: dict[str, np.ndarray], branch_id: int) -> tuple[int, float, int]:
    idx = np.where(arrays["cf_branch_id"] == branch_id)[0]
    if len(idx) == 0:
        return (999, 999.0, int(branch_id))
    i = int(idx[0])
    return (
        int(arrays["cf_bin_id"][i]),
        float(arrays["cf_sampled_steer"][i]),
        int(branch_id),
    )


def select_branch_ids_for_anchor(
    arrays: dict[str, np.ndarray],
    *,
    anchor_id: int,
    max_branches: int | None,
) -> list[int]:
    """Return sorted branch ids for one anchor."""
    mask = arrays["cf_anchor_id"] == anchor_id
    branch_ids = np.unique(arrays["cf_branch_id"][mask].astype(np.int64))
    branch_ids = sorted([int(b) for b in branch_ids], key=lambda b: branch_sort_key(arrays, b))

    if max_branches is not None:
        branch_ids = branch_ids[:max_branches]
    return branch_ids


def indices_for_branch(
    arrays: dict[str, np.ndarray],
    *,
    branch_id: int,
) -> np.ndarray:
    """Return frame indices for one branch sorted by cf_branch_step."""
    idx = np.where(arrays["cf_branch_id"] == branch_id)[0]
    if len(idx) == 0:
        raise ValueError(f"No frames found for branch_id={branch_id}")
    order = np.argsort(arrays["cf_branch_step"][idx])
    return idx[order]


def split_context_branch_indices(
    arrays: dict[str, np.ndarray],
    *,
    branch_id: int,
    branch_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return context indices and branch indices for one branch."""
    idx = indices_for_branch(arrays, branch_id=branch_id)
    steps = arrays["cf_branch_step"][idx]

    context_idx = idx[steps <= 0]
    branch_idx = idx[(steps >= 1) & (steps <= branch_horizon)]

    context_idx = context_idx[np.argsort(arrays["cf_branch_step"][context_idx])]
    branch_idx = branch_idx[np.argsort(arrays["cf_branch_step"][branch_idx])]

    if len(context_idx) == 0:
        raise ValueError(f"Branch {branch_id} has no context frames.")
    if len(branch_idx) != branch_horizon:
        raise ValueError(
            f"Branch {branch_id} branch length mismatch: "
            f"expected {branch_horizon}, got {len(branch_idx)}"
        )

    return context_idx, branch_idx


# ---------------------------------------------------------------------------
# Model rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_one_branch(
    *,
    agent: DeepAIFAgent,
    images_h5: h5py.Dataset,
    arrays: dict[str, np.ndarray],
    context_idx: np.ndarray,
    branch_idx: np.ndarray,
    deterministic: bool,
) -> dict[str, Any]:
    """Posterior-update through context and prior-rollout branch actions."""
    wm = agent.world_model
    device = agent._device
    wm.eval()

    prev_state = wm.rssm.initial(1, device)

    # Context posterior updates.
    for frame_idx in context_idx:
        img_raw = torch.from_numpy(images_h5[int(frame_idx)]).unsqueeze(0).to(device)
        state = torch.from_numpy(arrays["states"][int(frame_idx)]).unsqueeze(0).float().to(device)
        action = torch.from_numpy(arrays["actions"][int(frame_idx)]).unsqueeze(0).float().to(device)

        img_model = wm.preprocess_image(img_raw)
        embed = wm.encoder(img_model, state)
        post, _ = wm.rssm.obs_step(prev_state, action, embed)

        if deterministic:
            post = deterministic_state(post)

        prev_state = type(post)(*[x.detach() for x in post])

    anchor_model_img = wm.preprocess_image(
        torch.from_numpy(images_h5[int(context_idx[-1])]).unsqueeze(0).to(device)
    )[0]

    pred_imgs: list[torch.Tensor] = []
    gt_imgs: list[torch.Tensor] = []
    pred_states: list[np.ndarray] = []

    prior_state = prev_state
    for frame_idx in branch_idx:
        action = torch.from_numpy(arrays["actions"][int(frame_idx)]).unsqueeze(0).float().to(device)
        prior_state = wm.rssm.img_step(prior_state, action)

        if deterministic:
            prior_state = deterministic_state(prior_state)

        decoded = wm.decode_obs(prior_state)[0]
        pred_imgs.append(decoded.detach().cpu())

        gt_raw = torch.from_numpy(images_h5[int(frame_idx)]).unsqueeze(0).to(device)
        gt_model = wm.preprocess_image(gt_raw)[0]
        gt_imgs.append(gt_model.detach().cpu())

        feat = wm.rssm.get_feat(prior_state)
        pred_state = wm.state_decoder(feat)[0].detach().float().cpu().numpy()
        pred_states.append(pred_state)

        prior_state = type(prior_state)(*[x.detach() for x in prior_state])

    return {
        "anchor_img": anchor_model_img.detach().cpu(),
        "gt_imgs": gt_imgs,
        "pred_imgs": pred_imgs,
        "pred_states": pred_states,
    }


def compute_branch_metrics(
    *,
    pred_imgs: list[torch.Tensor],
    gt_imgs: list[torch.Tensor],
    pred_states: list[np.ndarray],
    gt_states: np.ndarray,
) -> dict[str, float]:
    """Compute simple rollout metrics for one branch."""
    pred_stack = torch.stack(pred_imgs, dim=0).float()
    gt_stack = torch.stack(gt_imgs, dim=0).float()

    img_mse_by_t = ((pred_stack - gt_stack) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    img_mae_by_t = (pred_stack - gt_stack).abs().mean(dim=(1, 2, 3)).cpu().numpy()

    pred_state_arr = np.stack(pred_states).astype(np.float32)
    gt_state_arr = gt_states.astype(np.float32)

    state_mse_by_t = ((pred_state_arr - gt_state_arr) ** 2).mean(axis=1)
    cte_mae_by_t = np.abs(pred_state_arr[:, 3] - gt_state_arr[:, 3])
    heading_mae_by_t = np.abs(pred_state_arr[:, 2] - gt_state_arr[:, 2])

    return {
        "img_mse_mean": float(np.mean(img_mse_by_t)),
        "img_mse_final": float(img_mse_by_t[-1]),
        "img_mae_mean": float(np.mean(img_mae_by_t)),
        "img_mae_final": float(img_mae_by_t[-1]),
        "state_mse_mean": float(np.mean(state_mse_by_t)),
        "state_mse_final": float(state_mse_by_t[-1]),
        "cte_mae_mean": float(np.mean(cte_mae_by_t)),
        "cte_mae_final": float(cte_mae_by_t[-1]),
        "heading_mae_mean": float(np.mean(heading_mae_by_t)),
        "heading_mae_final": float(heading_mae_by_t[-1]),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def branch_label(arrays: dict[str, np.ndarray], branch_id: int, row_type: str) -> str:
    """Create row label."""
    idx = np.where(arrays["cf_branch_id"] == branch_id)[0]
    if len(idx) == 0:
        return row_type

    i = int(idx[0])
    bin_id = int(arrays["cf_bin_id"][i])
    steer = float(arrays["cf_sampled_steer"][i])

    flags = []
    inv = bool(safe_get_frame_value(arrays, "cf_branch_lane_invasion_10", i, False))
    col = bool(safe_get_frame_value(arrays, "cf_branch_collision_10", i, False))
    max_cte = float(safe_get_frame_value(arrays, "cf_max_abs_cte_delta_10", i, np.nan))

    if inv:
        flags.append("LI")
    if col:
        flags.append("COL")
    if not flags:
        flags.append("OK")

    return f"{row_type} b{bin_id} s={steer:+.3f} {'/'.join(flags)} cte={max_cte:.2f}"


def save_anchor_grid(
    *,
    anchor_id: int,
    task_label: int,
    branch_results: list[dict[str, Any]],
    steps_to_show: list[int],
    output_path: Path,
    tile_size: int,
    label_width: int,
    header_height: int,
    row_gap: int,
    col_gap: int,
) -> None:
    """Save one GT-vs-pred rollout grid for an anchor."""
    font_small = get_font(11)
    font_header = get_font(13)

    columns = ["anchor"] + [f"+{s}" for s in steps_to_show]
    n_cols = len(columns)
    n_rows = len(branch_results) * 2

    width = label_width + n_cols * tile_size + (n_cols - 1) * col_gap
    height = header_height + n_rows * tile_size + (n_rows - 1) * row_gap

    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    mode_name = "clean" if task_label == 0 else "obstacle"
    title = f"anchor={anchor_id} mode={mode_name} branches={len(branch_results)}"
    draw.text((8, 4), title, fill=(0, 0, 0), font=font_header)

    for c, label in enumerate(columns):
        x = label_width + c * (tile_size + col_gap)
        draw.text((x + 4, header_height - 18), label, fill=(0, 0, 0), font=font_small)

    row = 0
    for result in branch_results:
        anchor_img = result["anchor_img"]
        gt_imgs = result["gt_imgs"]
        pred_imgs = result["pred_imgs"]
        gt_label = result["gt_label"]
        pred_label = result["pred_label"]

        for row_label, imgs in [
            (gt_label, gt_imgs),
            (pred_label, pred_imgs),
        ]:
            y = header_height + row * (tile_size + row_gap)
            draw.text((6, y + 4), row_label, fill=(0, 0, 0), font=font_small)

            x0 = label_width
            canvas.paste(resize_tile(anchor_img, tile_size), (x0, y))

            for c, step in enumerate(steps_to_show, start=1):
                x = label_width + c * (tile_size + col_gap)
                img_idx = step - 1
                if img_idx < 0 or img_idx >= len(imgs):
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(210, 210, 210))
                    draw.text((x + 4, y + 4), "missing", fill=(0, 0, 0), font=font_small)
                    continue
                canvas.paste(resize_tile(imgs[img_idx], tile_size), (x, y))

            row += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rollout metric rows."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual v6 rollout grid.")
    parser.add_argument("--config", required=True, help="Config yaml path.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--data", required=True, help="Counterfactual v6 HDF5 data.")
    parser.add_argument("--output_dir", required=True, help="Output directory for rollout grids.")
    parser.add_argument("--mode", choices=["all", "clean", "obstacle"], default="all")
    parser.add_argument("--anchor_ids", default=None, help="Optional comma-separated anchor ids.")
    parser.add_argument("--max_anchors", type=int, default=4)
    parser.add_argument("--max_branches", type=int, default=None, help="Limit branches per anchor.")
    parser.add_argument("--steps", default="1,3,5,7,9,11,13,15")
    parser.add_argument("--branch_horizon", type=int, default=None, help="Defaults to HDF5 attr branch_horizon.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tile_size", type=int, default=96)
    parser.add_argument("--label_width", type=int, default=270)
    parser.add_argument("--header_height", type=int, default=48)
    parser.add_argument("--row_gap", type=int, default=3)
    parser.add_argument("--col_gap", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Use sampled stochastic states instead of mean states.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_to_show = parse_int_list(args.steps)
    if steps_to_show is None or len(steps_to_show) == 0:
        raise ValueError("--steps must contain at least one step index.")

    arrays, attrs = load_metadata(args.data)
    branch_horizon = args.branch_horizon
    if branch_horizon is None:
        branch_horizon = int(attrs.get("branch_horizon", max(steps_to_show)))

    if max(steps_to_show) > branch_horizon:
        raise ValueError(
            f"Requested step {max(steps_to_show)} but branch_horizon={branch_horizon}"
        )

    selected_anchors = select_anchor_ids(
        arrays,
        mode=args.mode,
        anchor_ids=parse_int_list(args.anchor_ids),
        max_anchors=args.max_anchors,
    )

    cfg = Config.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device

    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)
    agent.world_model.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print("=" * 100)
    print("Counterfactual rollout evaluation")
    print(f"config: {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {ckpt.get('epoch')}")
    print(f"checkpoint best_epoch: {ckpt.get('best_epoch')}")
    print(f"checkpoint best_loss: {ckpt.get('best_loss')}")
    print(f"data: {args.data}")
    print(f"output_dir: {output_dir}")
    print(f"model type: {getattr(cfg.model, 'world_model_type', 'rssm')}")
    print(f"selected anchors: {selected_anchors}")
    print(f"branch_horizon: {branch_horizon}")
    print(f"steps_to_show: {steps_to_show}")
    print(f"deterministic rollout: {not args.stochastic}")
    print("=" * 100)

    metric_rows: list[dict[str, Any]] = []

    with h5py.File(args.data, "r") as f:
        images_h5 = f["images"]

        for anchor_id in selected_anchors:
            anchor_mask = arrays["cf_anchor_id"] == anchor_id
            labels = arrays["task_labels"][anchor_mask]
            task_label = int(np.round(np.mean(labels))) if labels.size else 0
            mode_name = "clean" if task_label == 0 else "obstacle"

            branch_ids = select_branch_ids_for_anchor(
                arrays,
                anchor_id=anchor_id,
                max_branches=args.max_branches,
            )

            branch_results: list[dict[str, Any]] = []

            for branch_id in branch_ids:
                context_idx, branch_idx = split_context_branch_indices(
                    arrays,
                    branch_id=branch_id,
                    branch_horizon=branch_horizon,
                )

                result = rollout_one_branch(
                    agent=agent,
                    images_h5=images_h5,
                    arrays=arrays,
                    context_idx=context_idx,
                    branch_idx=branch_idx,
                    deterministic=not args.stochastic,
                )

                gt_states = arrays["states"][branch_idx]
                metrics = compute_branch_metrics(
                    pred_imgs=result["pred_imgs"],
                    gt_imgs=result["gt_imgs"],
                    pred_states=result["pred_states"],
                    gt_states=gt_states,
                )

                first_idx = int(branch_idx[0])
                row = {
                    "anchor_id": int(anchor_id),
                    "branch_id": int(branch_id),
                    "task_label": int(task_label),
                    "bin_id": int(arrays["cf_bin_id"][first_idx]),
                    "sampled_steer": float(arrays["cf_sampled_steer"][first_idx]),
                    **metrics,
                }
                metric_rows.append(row)

                gt_label = branch_label(arrays, branch_id, "GT")
                pred_label = (
                    f"Pred mseF={metrics['img_mse_final']:.4f} "
                    f"cteF={metrics['cte_mae_final']:.3f}"
                )

                branch_results.append({
                    "branch_id": branch_id,
                    "anchor_img": result["anchor_img"],
                    "gt_imgs": result["gt_imgs"],
                    "pred_imgs": result["pred_imgs"],
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "metrics": metrics,
                })

            out_path = output_dir / f"rollout_anchor_{anchor_id:04d}_{mode_name}.png"
            save_anchor_grid(
                anchor_id=anchor_id,
                task_label=task_label,
                branch_results=branch_results,
                steps_to_show=steps_to_show,
                output_path=out_path,
                tile_size=args.tile_size,
                label_width=args.label_width,
                header_height=args.header_height,
                row_gap=args.row_gap,
                col_gap=args.col_gap,
            )
            print(f"saved {out_path} | branches={len(branch_results)}")

    metrics_path = output_dir / "rollout_metrics.csv"
    write_metrics_csv(metrics_path, metric_rows)

    if metric_rows:
        print("=" * 100)
        print(f"metrics: {metrics_path}")
        for key in ["img_mse_mean", "img_mse_final", "state_mse_mean", "cte_mae_final", "heading_mae_final"]:
            vals = np.array([float(r[key]) for r in metric_rows], dtype=np.float32)
            print(f"{key:20s}: mean={vals.mean():.6f} std={vals.std():.6f}")

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()

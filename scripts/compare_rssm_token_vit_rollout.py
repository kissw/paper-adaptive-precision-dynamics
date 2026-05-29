"""Compare RSSM vs TokenViT transition prior rollout visualisation.

Rolls out both world models from the same context frames using only
actions (no future GT observations).  Produces a 5-row grid:

  Row 1  GT future frames
  Row 2  RSSM prior rollout
  Row 3  RSSM per-pixel L1 error (×error_scale, clipped to [0,1])
  Row 4  TokenViT prior rollout
  Row 5  TokenViT per-pixel L1 error

Columns = horizon steps t+1 … t+H.

Usage:
    ~/.local/bin/uv run python scripts/compare_rssm_token_vit_rollout.py \\
        --rssm_checkpoint runs/rssm_default_20260527_235402/checkpoints/best.pt \\
        --rssm_config     configs/experiment/task_b_v5.yaml \\
        --token_vit_checkpoint runs/token_vit_20260528_000427/checkpoints/best.pt \\
        --token_vit_config     configs/experiment/token_vit.yaml \\
        --data data/expert_data_v4.h5 \\
        --start_index 1000 --context_len 5 --horizon 15 \\
        --output_dir rollout_compare_obstacle --case_name obstacle_1000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False

from active_inference.config import Config
from active_inference.agent import WorldModel
from active_inference.utils.transforms import crop_road as _crop_road_transform


def ssim_fn(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0, channel_axis: int = 2) -> float:
    """Thin wrapper: use skimage if available, else return NaN."""
    if _SSIM_AVAILABLE:
        return float(_skimage_ssim(gt, pred, data_range=data_range, channel_axis=channel_axis))
    return float("nan")


# ---------------------------------------------------------------------------
# crop_road helpers
# ---------------------------------------------------------------------------

def get_crop_road_from_config(cfg_path: str) -> bool:
    """Return the encoder.crop_road flag from a YAML config (default False)."""
    cfg = Config.from_yaml(cfg_path)
    return bool(getattr(getattr(cfg, "encoder", None), "crop_road", False))


def apply_crop_road(img: torch.Tensor) -> torch.Tensor:
    """Apply crop_road transform to a (C,H,W) or (B,C,H,W) tensor."""
    return _crop_road_transform(img)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_uint8(x: torch.Tensor, input_range: str = "auto") -> np.ndarray:
    """Convert (C,H,W) or (H,W,C) float tensor/array to (H,W,3) uint8.

    Handles [-0.5, 0.5] (stored range in expert_data) and [0,1].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()
    else:
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.transpose(1, 2, 0)

    if input_range == "auto":
        vmin = x.min()
        if vmin < -0.1:
            # Likely [-0.5, 0.5]
            x = x + 0.5
        # else assume [0, 1]
    elif input_range == "[-0.5,0.5]":
        x = x + 0.5

    x = np.clip(x, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """gt and pred are (H, W, 3) float32 in [0, 1]."""
    mse = float(np.mean((gt - pred) ** 2))
    psnr = psnr_from_mse(mse)
    # SSIM over all channels
    ssim = float(ssim_fn(gt, pred, data_range=1.0, channel_axis=2))
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


# ---------------------------------------------------------------------------
# World model rollout
# ---------------------------------------------------------------------------

def build_world_model(cfg_path: str, ckpt_path: str, device: torch.device) -> WorldModel:
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    return wm


def prepare_gt_images(
    images: torch.Tensor,
    start: int,
    context_len: int,
    horizon: int,
    crop_road: bool,
) -> list[np.ndarray]:
    """Return `horizon` GT display images, optionally crop_road-processed."""
    gt_list = []
    for h in range(horizon):
        t = start + context_len + h
        img = images[t]                               # (3, H, W)
        if crop_road:
            img = apply_crop_road(img)
        gt_list.append(to_uint8(img))
    return gt_list


def rollout_model(
    wm: WorldModel,
    images: torch.Tensor,   # (T_total, 3, H, W)  float, on CPU
    states: torch.Tensor,   # (T_total, S)
    actions: torch.Tensor,  # (T_total, A)  — actions[i] transitions i→i+1
    start: int,
    context_len: int,
    horizon: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Run context obs_step then horizon img_step.

    Returns a list of `horizon` decoded image tensors, each (3, H, W) on CPU.
    Future GT observations are never fed into obs_step after context.
    """
    T = images.shape[0]
    end_ctx  = start + context_len   # first rollout step is this index
    end_roll = end_ctx + horizon

    assert end_roll <= T, (
        f"start({start}) + context_len({context_len}) + horizon({horizon}) = "
        f"{end_roll} > data length {T}"
    )

    with torch.no_grad():
        # ---- Context phase: posterior obs_step ----
        state = wm.rssm.initial(1, device)
        prev_act = torch.zeros(1, actions.shape[-1], device=device)

        for i in range(context_len):
            t = start + i
            img = images[t : t + 1].to(device)
            st  = states[t : t + 1].to(device)
            embed = wm.encoder(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            state = type(post)(*[x.detach() for x in post])
            # action at time t moves us to time t+1
            if t < T - 1:
                prev_act = actions[t : t + 1].to(device)
            else:
                prev_act = torch.zeros_like(prev_act)

        # ---- Rollout phase: prior img_step only ----
        rollout_state = state
        preds: list[torch.Tensor] = []

        for h in range(horizon):
            t = end_ctx + h - 1   # action at this index transitions → t+1
            act = actions[t : t + 1].to(device) if t < T - 1 else torch.zeros(1, actions.shape[-1], device=device)
            rollout_state = wm.rssm.img_step(rollout_state, act)
            pred_img = wm.decode_obs(rollout_state)   # (1, 3, H, W)
            preds.append(pred_img.squeeze(0).cpu())

    return preds


# ---------------------------------------------------------------------------
# Grid visualisation
# ---------------------------------------------------------------------------

def make_grid(
    gt_rssm_imgs: list[np.ndarray],  # GT as seen by RSSM (crop_road if True)
    rssm_imgs: list[np.ndarray],
    rssm_errs: list[np.ndarray],
    gt_tvit_imgs: list[np.ndarray],  # GT as seen by TokenViT
    tvit_imgs: list[np.ndarray],
    tvit_errs: list[np.ndarray],
    title: str,
    error_scale: float,
    dpi: int = 120,
    crop_road_rssm: bool = False,
    crop_road_tvit: bool = False,
) -> plt.Figure:
    horizon = len(gt_rssm_imgs)

    # Decide row count: add a "GT raw" row when crop_road is active on either model
    show_raw_gt = crop_road_rssm or crop_road_tvit
    n_rows = 6 if show_raw_gt else 5
    n_cols = horizon

    fw = max(n_cols * 1.2, 8)
    fh = n_rows * 1.4 + 1.2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh), dpi=dpi)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    if show_raw_gt:
        rssm_gt_label = "GT (crop_road)" if crop_road_rssm else "GT (full)"
        tvit_gt_label = "GT (crop_road)" if crop_road_tvit else "GT (full)"
        row_labels = [
            rssm_gt_label,
            "RSSM rollout",
            "RSSM L1 error",
            tvit_gt_label,
            "TokenViT rollout",
            "TokenViT L1 error",
        ]
        row_data = [
            gt_rssm_imgs, rssm_imgs, rssm_errs,
            gt_tvit_imgs, tvit_imgs, tvit_errs,
        ]
        is_error = [False, False, True, False, False, True]
    else:
        row_labels = [
            "GT future",
            "RSSM rollout",
            "RSSM L1 error",
            "TokenViT rollout",
            "TokenViT L1 error",
        ]
        row_data = [gt_rssm_imgs, rssm_imgs, rssm_errs, tvit_imgs, tvit_errs]
        is_error = [False, False, True, False, True]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            img = row_data[r][c]
            if is_error[r]:
                ax.imshow(img, cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"+{c+1}", fontsize=7)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=6, rotation=90, labelpad=2)

    fig.suptitle(title, fontsize=7, y=1.01, wrap=True)
    fig.tight_layout(pad=0.3)
    return fig


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------

def run_case(
    wm_rssm: WorldModel,
    wm_tvit: WorldModel,
    images: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    start: int,
    context_len: int,
    horizon: int,
    device: torch.device,
    case_name: str,
    output_dir: Path,
    rssm_ckpt_name: str,
    tvit_ckpt_name: str,
    data_name: str,
    error_scale: float,
    dpi: int,
    save_gif: bool,
    crop_road_rssm: bool = False,
    crop_road_tvit: bool = False,
) -> list[dict]:
    """Run one comparison case. Returns list of metric dicts."""

    # ---- GT future (model-matched; crop_road applied per-model) ----
    gt_rssm_list = prepare_gt_images(images, start, context_len, horizon, crop_road=crop_road_rssm)
    gt_tvit_list = prepare_gt_images(images, start, context_len, horizon, crop_road=crop_road_tvit)

    # ---- RSSM rollout ----
    t0 = time.time()
    rssm_preds = rollout_model(
        wm_rssm, images, states, actions, start, context_len, horizon, device,
    )
    print(f"  RSSM rollout done in {time.time()-t0:.1f}s")

    # ---- TokenViT rollout ----
    t0 = time.time()
    tvit_preds = rollout_model(
        wm_tvit, images, states, actions, start, context_len, horizon, device,
    )
    print(f"  TokenViT rollout done in {time.time()-t0:.1f}s")

    # Log min/max for diagnostics
    gt_t = images[start + context_len]
    rssm_t = rssm_preds[0]
    tvit_t = tvit_preds[0]
    print(f"  Image range diagnostic (step +1):")
    print(f"    GT    min={gt_t.min():.3f} max={gt_t.max():.3f}")
    print(f"    RSSM  min={rssm_t.min():.3f} max={rssm_t.max():.3f}")
    print(f"    TVit  min={tvit_t.min():.3f} max={tvit_t.max():.3f}")

    # ---- Compute error maps and metrics ----
    rssm_imgs_u8, rssm_errs, tvit_imgs_u8, tvit_errs = [], [], [], []
    metrics_rows: list[dict] = []

    for h in range(horizon):
        rssm_gt_u8 = gt_rssm_list[h]
        tvit_gt_u8 = gt_tvit_list[h]
        rssm_u8    = to_uint8(rssm_preds[h])
        tvit_u8    = to_uint8(tvit_preds[h])

        rssm_gt_f = rssm_gt_u8.astype(np.float32) / 255.0
        tvit_gt_f = tvit_gt_u8.astype(np.float32) / 255.0
        rssm_f    = rssm_u8.astype(np.float32) / 255.0
        tvit_f    = tvit_u8.astype(np.float32) / 255.0

        # L1 error against model-matched GT
        rssm_err = np.abs(rssm_gt_f - rssm_f).mean(axis=2)   # (H,W)
        tvit_err = np.abs(tvit_gt_f - tvit_f).mean(axis=2)

        rssm_errs.append(np.clip(rssm_err * error_scale, 0.0, 1.0))
        tvit_errs.append(np.clip(tvit_err * error_scale, 0.0, 1.0))

        rssm_imgs_u8.append(rssm_u8)
        tvit_imgs_u8.append(tvit_u8)

        # Metrics against model-matched GT
        m_rssm = compute_metrics(rssm_gt_f, rssm_f)
        m_tvit = compute_metrics(tvit_gt_f, tvit_f)
        metrics_rows.append({
            "case_name": case_name, "model": "RSSM", "horizon_step": h + 1,
            "mse": m_rssm["mse"], "psnr": m_rssm["psnr"], "ssim": m_rssm["ssim"],
        })
        metrics_rows.append({
            "case_name": case_name, "model": "TokenViT", "horizon_step": h + 1,
            "mse": m_tvit["mse"], "psnr": m_tvit["psnr"], "ssim": m_tvit["ssim"],
        })

    # ---- Grid ----
    title = (
        f"data={data_name}  start={start}  ctx={context_len}  H={horizon}\n"
        f"RSSM={rssm_ckpt_name}  TokenViT={tvit_ckpt_name}"
    )
    fig = make_grid(
        gt_rssm_imgs=gt_rssm_list,
        rssm_imgs=rssm_imgs_u8,
        rssm_errs=rssm_errs,
        gt_tvit_imgs=gt_tvit_list,
        tvit_imgs=tvit_imgs_u8,
        tvit_errs=tvit_errs,
        title=title, error_scale=error_scale, dpi=dpi,
        crop_road_rssm=crop_road_rssm,
        crop_road_tvit=crop_road_tvit,
    )
    grid_path = output_dir / f"{case_name}_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved -> {grid_path}")

    # ---- Optional GIF ----
    if save_gif:
        try:
            from PIL import Image as PILImage
            frames_rssm = [PILImage.fromarray(img) for img in rssm_imgs_u8]
            gif_path = output_dir / f"{case_name}_rssm.gif"
            frames_rssm[0].save(
                gif_path, save_all=True, append_images=frames_rssm[1:],
                duration=200, loop=0,
            )
            frames_tvit = [PILImage.fromarray(img) for img in tvit_imgs_u8]
            gif_path2 = output_dir / f"{case_name}_tokenvit.gif"
            frames_tvit[0].save(
                gif_path2, save_all=True, append_images=frames_tvit[1:],
                duration=200, loop=0,
            )
            print(f"  GIFs saved -> {gif_path}, {gif_path2}")
        except ImportError:
            print("  PIL not available; skipping GIF export.")

    return metrics_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RSSM vs TokenViT prior rollout",
    )
    # Models
    parser.add_argument("--rssm_checkpoint",       required=True)
    parser.add_argument("--rssm_config",           required=True)
    parser.add_argument("--token_vit_checkpoint",  required=True)
    parser.add_argument("--token_vit_config",      required=True)
    # Data
    parser.add_argument("--data",                  required=True)
    parser.add_argument("--start_index",           type=int, required=True)
    parser.add_argument("--context_len",           type=int, default=5)
    parser.add_argument("--horizon",               type=int, default=15)
    # Output
    parser.add_argument("--output_dir",            default="rollout_compare")
    parser.add_argument("--case_name",             default="case")
    parser.add_argument("--metrics_csv",           default="rollout_metrics.csv")
    parser.add_argument("--save_gif",              action="store_true", default=False)
    # Multi-case
    parser.add_argument("--num_cases",             type=int, default=1)
    parser.add_argument("--stride",                type=int, default=100)
    # Display
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--dpi",                   type=int, default=120)
    parser.add_argument("--error_scale",           type=float, default=4.0)
    # crop_road: force both models to use crop_road (otherwise auto-read from configs)
    parser.add_argument("--crop_road",             action="store_true", default=False,
                        help="Force crop_road for both models. Default: auto-read from each config.")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print(f"Loading data: {args.data}")
    with h5py.File(args.data, "r") as f:
        img_key = "images" if "images" in f else "frames"
        images  = torch.tensor(np.array(f[img_key]),  dtype=torch.float32)
        states  = torch.tensor(np.array(f["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(f["actions"]),dtype=torch.float32)

    T = images.shape[0]
    A = actions.shape[-1]
    data_name = Path(args.data).name

    print(f"  images:  {images.shape}  range [{images.min():.3f}, {images.max():.3f}]")
    print(f"  states:  {states.shape}")
    print(f"  actions: {actions.shape}")

    # Pad actions to match image length if needed (safety)
    if actions.shape[0] < T:
        pad = torch.zeros(T - actions.shape[0], A)
        actions = torch.cat([actions, pad], dim=0)

    # ---- Load world models ----
    print(f"\nLoading RSSM: {args.rssm_checkpoint}")
    wm_rssm = build_world_model(args.rssm_config, args.rssm_checkpoint, device)

    print(f"Loading TokenViT: {args.token_vit_checkpoint}")
    wm_tvit = build_world_model(args.token_vit_config, args.token_vit_checkpoint, device)

    rssm_name = Path(args.rssm_checkpoint).parts[-3] if len(Path(args.rssm_checkpoint).parts) >= 3 else Path(args.rssm_checkpoint).name
    tvit_name  = Path(args.token_vit_checkpoint).parts[-3] if len(Path(args.token_vit_checkpoint).parts) >= 3 else Path(args.token_vit_checkpoint).name

    print(f"\n  RSSM type     : {wm_rssm._wm_type}")
    print(f"  TokenViT type : {wm_tvit._wm_type}")

    # ---- Resolve crop_road flags ----
    crop_road_rssm = args.crop_road or get_crop_road_from_config(args.rssm_config)
    crop_road_tvit = args.crop_road or get_crop_road_from_config(args.token_vit_config)
    print(f"  crop_road RSSM    : {crop_road_rssm}")
    print(f"  crop_road TokenViT: {crop_road_tvit}")

    # ---- Multi-case loop ----
    all_metrics: list[dict] = []

    for idx in range(args.num_cases):
        start = args.start_index + idx * args.stride
        end   = start + args.context_len + args.horizon

        if end > T:
            print(f"  Case {idx}: start={start} end={end} > T={T}, skipping.")
            continue

        if args.num_cases == 1:
            case_name = args.case_name
        else:
            case_name = f"{args.case_name}_idx{idx:03d}_start{start}"

        print(f"\n--- Case {idx}: {case_name}  start={start} ---")

        rows = run_case(
            wm_rssm=wm_rssm,
            wm_tvit=wm_tvit,
            images=images,
            states=states,
            actions=actions,
            start=start,
            context_len=args.context_len,
            horizon=args.horizon,
            device=device,
            case_name=case_name,
            output_dir=output_dir,
            rssm_ckpt_name=rssm_name,
            tvit_ckpt_name=tvit_name,
            data_name=data_name,
            error_scale=args.error_scale,
            dpi=args.dpi,
            save_gif=args.save_gif,
            crop_road_rssm=crop_road_rssm,
            crop_road_tvit=crop_road_tvit,
        )
        all_metrics.extend(rows)

        # Print per-horizon summary
        rssm_rows = [r for r in rows if r["model"] == "RSSM"]
        tvit_rows  = [r for r in rows if r["model"] == "TokenViT"]
        print(f"\n  Horizon summary (RSSM vs TokenViT):")
        print(f"  {'step':>4}  {'RSSM PSNR':>10}  {'TVit PSNR':>10}  {'RSSM SSIM':>10}  {'TVit SSIM':>10}")
        for rr, tr in zip(rssm_rows, tvit_rows):
            print(
                f"  {rr['horizon_step']:>4}  "
                f"{rr['psnr']:>10.2f}  {tr['psnr']:>10.2f}  "
                f"{rr['ssim']:>10.4f}  {tr['ssim']:>10.4f}"
            )

    # ---- Save metrics CSV ----
    if all_metrics:
        csv_path = output_dir / args.metrics_csv
        fieldnames = ["case_name", "model", "horizon_step", "mse", "psnr", "ssim"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nMetrics saved -> {csv_path}")


if __name__ == "__main__":
    main()

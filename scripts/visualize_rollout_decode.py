#!/usr/bin/env python3
"""Visualize prior rollout decode quality across model variants.

For each model, encodes a starting obstacle frame to obtain an initial
posterior, then rolls out h steps using img_step (prior only) with the
ground-truth actions.  Decoded images at each horizon are compared with
the actual future frames (GT row).

Grid layout:
    rows : [GT, rssm_off, rssm_on, vit_off, vit_on, vit_a1]
    cols : horizons (e.g. 1, 3, 5, 7, 10, 15, 20)

Usage:
    uv run python scripts/visualize_rollout_decode.py \\
        --eval_data data/pp_gap_eval_v5.h5 \\
        --start_frames 0,30 \\
        --horizons 1,3,5,7,10,15,20 \\
        --output outputs/rollout_decode_grid.png

Custom runs (optional JSON):
    uv run python ... --runs_json path/to/runs.json
    where runs.json is a list of {"name":"...", "config":"...", "checkpoint":"...", "overrides":[...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent))
from pp_gap_v5 import build_world_model  # config-aware, overrides-aware


# ─────────────────────────────────────────────────────────────────────────────
# Default model specifications
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RUNS = [
    {
        "name": "rssm_off",
        "config": "configs/default.yaml",
        "checkpoint": "outputs/v6_rssm_off/checkpoints/best.pt",
        "overrides": [],
    },
    {
        "name": "rssm_on",
        "config": "configs/default.yaml",
        "checkpoint": "outputs/v6_rssm_on/checkpoints/best.pt",
        "overrides": [],
    },
    {
        "name": "vit_off",
        "config": "configs/experiment/token_vit.yaml",
        "checkpoint": "outputs/v6_vit_off/checkpoints/best.pt",
        "overrides": [],
    },
    {
        "name": "vit_on",
        "config": "configs/experiment/token_vit.yaml",
        "checkpoint": "outputs/v6_vit_on/checkpoints/best.pt",
        "overrides": [],
    },
    {
        "name": "vit_a1",
        "config": "configs/experiment/token_vit.yaml",
        "checkpoint": "outputs/v6_vit_a1/checkpoints/best.pt",
        "overrides": ["token_vit.use_action_warp=true"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_group(path: str, group: str):
    with h5py.File(path, "r") as f:
        grp = f[group]
        images     = torch.tensor(np.array(grp["images"]),     dtype=torch.float32)
        states     = torch.tensor(np.array(grp["states"]),     dtype=torch.float32)
        actions    = torch.tensor(np.array(grp["actions"]),    dtype=torch.float32)
        episode_ids = np.array(grp["episode_ids"])
    return images, states, actions, episode_ids


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic state helper
# ─────────────────────────────────────────────────────────────────────────────

def make_deterministic(state):
    """Replace stoch with distribution mean for noise-free decoding.

    RSSM  : RSSMState(deter, stoch=mean, mean, std)
    ViT   : TokenRSSMState(deter, stoch=token_mean, mean, std, token_mean, token_std)
    """
    if hasattr(state, "token_mean"):          # TokenRSSMState (ViT)
        return type(state)(
            deter=state.deter,
            stoch=state.token_mean,           # replace sampled stoch with mean
            mean=state.mean,
            std=state.std,
            token_mean=state.token_mean,
            token_std=state.token_std,
        )
    else:                                      # RSSMState (RSSM)
        return type(state)(
            deter=state.deter,
            stoch=state.mean,                 # replace sampled stoch with mean
            mean=state.mean,
            std=state.std,
        )


@contextmanager
def deterministic_normal():
    """Patch Normal.rsample to return the mean, suppressing stochastic noise."""
    _orig = Normal.rsample

    def _det(self, sample_shape=()):
        return self.loc

    Normal.rsample = _det
    try:
        yield
    finally:
        Normal.rsample = _orig


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_and_decode(
    wm,
    images: Tensor,
    states: Tensor,
    actions: Tensor,
    global_start: int,
    horizons: list[int],
    device: torch.device,
    context_len: int = 5,
) -> dict[int, Tensor]:
    """Encode context frames, then roll out prior for each horizon.

    Returns {h: decoded_image (3, H, W)} for h in horizons.
    """
    ep_len = len(images)
    h_max = max(horizons)

    # ── Context encoding ──────────────────────────────────────────────────
    # Use up to context_len frames ending at global_start (within the slice).
    ctx_start = max(0, global_start - context_len + 1)
    state = wm.rssm.initial(1, device)
    action_dim = wm._cfg.cem.action_dim
    prev_act = torch.zeros(1, action_dim, device=device)

    with deterministic_normal():
        for t in range(ctx_start, global_start + 1):
            img = images[t:t+1].to(device)
            st  = states[t:t+1].to(device)
            act = actions[t:t+1].to(device)
            embed = wm.encode_obs(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            state   = type(post)(*[x.detach() for x in post])
            prev_act = act

    # Use posterior mean as clean starting point
    state = make_deterministic(state)

    # ── Prior rollout ─────────────────────────────────────────────────────
    decoded: dict[int, Tensor] = {}
    with deterministic_normal():
        for h in range(1, h_max + 1):
            act_idx = global_start + h
            if act_idx >= ep_len:
                break
            act = actions[act_idx:act_idx+1].to(device)
            state = wm.rssm.img_step(state, act)
            state = make_deterministic(state)
            if h in horizons:
                dec = wm.decode_obs(state)          # (1, 3, H, W)
                decoded[h] = dec.squeeze(0).cpu()

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# GT decoder reconstruction (upper bound row)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_gt_frames(
    wm,
    images: Tensor,
    states: Tensor,
    start_idx: int,
    horizons: list[int],
    device: torch.device,
) -> dict[int, Tensor]:
    """For each GT frame at start_idx+h, encode with fresh state then decode.

    Shows the decoder's reconstruction ceiling (posterior → decode)
    as opposed to the prior rollout rows which never see the future frames.
    """
    result: dict[int, Tensor] = {}
    action_dim = wm._cfg.cem.action_dim
    for h in horizons:
        idx = start_idx + h
        if idx >= len(images):
            break
        img = images[idx:idx+1].to(device)
        st  = states[idx:idx+1].to(device)
        # Fresh initial state + zero prev_action: 1-step reconstruction
        init_state = wm.rssm.initial(1, device)
        prev_act   = torch.zeros(1, action_dim, device=device)
        with deterministic_normal():
            embed      = wm.encode_obs(img, st)
            post, _    = wm.rssm.obs_step(init_state, prev_act, embed)
            post       = make_deterministic(post)
            dec        = wm.decode_obs(post)
        result[h] = dec.squeeze(0).cpu()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_display(img: Tensor) -> np.ndarray:
    """Convert (3, H, W) float tensor to (H, W, 3) uint8 for matplotlib.

    Assumes images are in approximately [-0.5, 0.5] or [0, 1] range.
    Applies +0.5 shift then clamp to [0, 1].
    """
    arr = img.detach().float()
    arr = (arr + 0.5).clamp(0.0, 1.0)
    return arr.permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Grid plot
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(
    gt_row: dict[int, Tensor],
    dec_row: dict[int, Tensor],
    model_rows: list[tuple[str, dict[int, Tensor]]],
    horizons: list[int],
    start_label: str,
    output_path: str,
):
    """Build comparison grid.

    Row order: GT | Dec | model_0 | model_1 | ...
    Col order: h=horizons[0] ... h=horizons[-1]
    Row labels are drawn as text on the left edge of each row's first cell.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available")
        return

    all_rows = [("GT", gt_row), ("Dec", dec_row)] + list(model_rows)
    n_rows = len(all_rows)
    n_cols = len(horizons)

    # Extra left column for row labels
    col_ratios = [0.55] + [1.0] * n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=((n_cols + 0.55) * 1.3, n_rows * 1.45),
        gridspec_kw={"width_ratios": col_ratios},
        dpi=120,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Separator row index: between Dec and first model
    SEPARATOR_AFTER = 1  # after "Dec"

    for r, (label, row_data) in enumerate(all_rows):
        # Label cell (column 0)
        lax = axes[r, 0]
        lax.axis("off")
        color = "#e8f4e8" if label == "GT" else "#e8eef8" if label == "Dec" else "#ffffff"
        lax.set_facecolor(color)
        lax.text(
            0.92, 0.5, label,
            transform=lax.transAxes,
            fontsize=8, fontweight="bold" if label in ("GT", "Dec") else "normal",
            ha="right", va="center",
        )

        # Draw a thin separator line below Dec row
        if r == SEPARATOR_AFTER:
            lax.axhline(y=0.02, color="#888", linewidth=1.0)

        # Image cells (columns 1..n_cols)
        for c, h in enumerate(horizons):
            ax = axes[r, c + 1]
            ax.axis("off")
            if r == SEPARATOR_AFTER:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if h in row_data:
                ax.imshow(to_display(row_data[h]), interpolation="nearest",
                          aspect="equal")
            else:
                ax.set_facecolor("#222222")
            if r == 0:
                ax.set_title(f"h={h}", fontsize=7.5, pad=3,
                             fontweight="bold")

    fig.suptitle(
        f"Prior rollout  ·  start {start_label}  ·  "
        "GT=actual | Dec=encoder→decoder | others=prior rollout",
        fontsize=8, y=1.005,
    )
    plt.tight_layout(pad=0.25, h_pad=0.15, w_pad=0.1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise prior rollout decode quality across model variants",
    )
    parser.add_argument(
        "--eval_data", default="data/pp_gap_eval_v5.h5",
        help="HDF5 with 'obstacle_visible' group",
    )
    parser.add_argument(
        "--group", default="obstacle_visible",
        help="HDF5 group to use (default: obstacle_visible)",
    )
    parser.add_argument(
        "--start_frames", default="0,30",
        help="Comma-separated list of frame indices within the group (e.g. 0,30,60)",
    )
    parser.add_argument(
        "--horizons", default="1,3,5,7,10,15,20",
        help="Comma-separated rollout horizons",
    )
    parser.add_argument(
        "--context_len", type=int, default=5,
        help="Frames used for context encoding before the start frame",
    )
    parser.add_argument(
        "--runs_json", default=None,
        help="Optional JSON file with list of model specs "
             "(name, config, checkpoint, overrides). "
             "Defaults to the 5 built-in model variants.",
    )
    parser.add_argument(
        "--output", default="outputs/rollout_decode_grid.png",
        help="Output PNG path (start frame index is appended for multiple frames)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    horizons    = [int(h) for h in args.horizons.split(",")]
    start_idxs  = [int(i) for i in args.start_frames.split(",")]
    device      = torch.device(args.device if torch.cuda.is_available() else "cpu")
    h_max       = max(horizons)

    # Load run specs
    runs = DEFAULT_RUNS
    if args.runs_json:
        with open(args.runs_json) as f:
            runs = json.load(f)

    # Filter to checkpoints that actually exist
    runs_valid = []
    for r in runs:
        if Path(r["checkpoint"]).exists():
            runs_valid.append(r)
        else:
            print(f"  SKIP {r['name']}: checkpoint not found → {r['checkpoint']}")
    if not runs_valid:
        print("ERROR: no valid checkpoints found")
        sys.exit(1)

    # Load eval data
    print(f"Loading {args.group} from {args.eval_data} …")
    images, states, actions, episode_ids = load_group(args.eval_data, args.group)
    print(f"  {len(images)} frames, {len(np.unique(episode_ids))} episodes")

    # Build all world models (load once, reuse for all start frames)
    print("\nBuilding models …")
    models: list[tuple[str, object]] = []
    for spec in runs_valid:
        print(f"  [{spec['name']}] {spec['checkpoint']}")
        wm, _ckpt, cfg = build_world_model(
            spec["config"],
            spec["checkpoint"],
            device,
            overrides=spec.get("overrides") or None,
        )
        # Store cfg on wm for action_dim access in rollout
        wm._cfg = cfg
        models.append((spec["name"], wm))
    print(f"  {len(models)} models loaded")

    # Process each start frame
    output_stem = Path(args.output).stem
    output_dir  = Path(args.output).parent
    output_suffix = Path(args.output).suffix or ".png"

    for start_idx in start_idxs:
        if start_idx >= len(images):
            print(f"  SKIP start_idx={start_idx}: beyond data length {len(images)}")
            continue

        ep = int(episode_ids[start_idx])
        ep_mask = episode_ids == ep
        ep_global_idxs = np.where(ep_mask)[0]
        ep_local_offset = int(np.where(ep_global_idxs == start_idx)[0][0])
        ep_remaining = len(ep_global_idxs) - ep_local_offset - 1

        # Ensure we have enough episode frames for the requested horizons
        available_h = min(h_max, ep_remaining)
        valid_horizons = [h for h in horizons if h <= available_h]
        if not valid_horizons:
            print(f"  SKIP start_idx={start_idx}: not enough episode frames for h>0")
            continue

        print(f"\nstart_idx={start_idx}  ep={ep}  "
              f"ep_remaining={ep_remaining}  valid_h={valid_horizons}")

        # GT row: actual future frames from the episode
        gt_row: dict[int, Tensor] = {}
        for h in valid_horizons:
            gt_idx = start_idx + h
            gt_row[h] = images[gt_idx].cpu()

        # Dec row: GT frames reconstructed through encoder → decoder (first model)
        first_wm = models[0][1]
        print(f"  [Dec] reconstructing GT via {models[0][0]} encoder …", end="", flush=True)
        dec_row = reconstruct_gt_frames(
            wm=first_wm,
            images=images,
            states=states,
            start_idx=start_idx,
            horizons=valid_horizons,
            device=device,
        )
        print(f" done ({len(dec_row)} horizons)")

        # Model rows: prior rollout decode
        model_rows: list[tuple[str, dict[int, Tensor]]] = []
        for name, wm in models:
            print(f"  [{name}] rolling out …", end="", flush=True)
            decoded = rollout_and_decode(
                wm=wm,
                images=images,
                states=states,
                actions=actions,
                global_start=start_idx,
                horizons=valid_horizons,
                device=device,
                context_len=args.context_len,
            )
            model_rows.append((name, decoded))
            print(f" done ({len(decoded)} horizons)")

        # Build and save grid
        out_path = str(output_dir / f"{output_stem}_idx{start_idx}{output_suffix}")
        build_grid(
            gt_row=gt_row,
            dec_row=dec_row,
            model_rows=model_rows,
            horizons=valid_horizons,
            start_label=f"frame {start_idx}  ep {ep}",
            output_path=out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

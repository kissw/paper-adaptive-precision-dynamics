#!/usr/bin/env python3
"""Rollout drift check: does the prior actually evolve the latent during rollout?

Motivation
----------
The obstacle probe shows RSSM staying ~0.87 accuracy even at h=20.  This could
mean either:
  (a) the prior genuinely propagates obstacle information forward, OR
  (b) the rollout barely changes the latent, so the probe just re-reads the
      frozen start posterior — an "information-carrying artifact".

This script distinguishes the two by measuring, per rollout step h:
  - drift_from_start : ||feat_h - feat_0||   (how far latent moved from start)
  - drift_step       : ||feat_h - feat_{h-1}|| (per-step change)
  - recon_mse        : MSE(decode(feat_h), GT_future_frame)  (predictive quality)

Interpretation:
  - RSSM drift ≈ 0  → rollout does not change latent → high probe = artifact.
  - ViT drift large  → latent genuinely evolves through rollout.
  - If RSSM recon_mse > ViT recon_mse (RSSM can't predict) while RSSM probe is
    higher → the probe metric is unsuitable (opposite conclusion).

Usage
-----
    uv run python scripts/rollout_drift_check.py \\
        --runs_json runs_rollout_recon.json \\
        --data data/expert_obstacle_v5.h5 \\
        --max_samples 200 --max_h 20 --context_len 5 \\
        --output_dir outputs/rollout_drift --worker 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _early_worker_limit(default: int = 10) -> int:
    """Cap BLAS/OpenMP threads before numpy/torch import (same as obstacle_probe)."""
    w = default
    if "--worker" in sys.argv:
        try:
            w = int(sys.argv[sys.argv.index("--worker") + 1])
        except (ValueError, IndexError):
            pass
    w = max(1, w)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(w)
    return w


_N_WORKERS = _early_worker_limit()

import h5py
import numpy as np
import torch

torch.set_num_threads(_N_WORKERS)

sys.path.insert(0, str(Path(__file__).parent))
from pp_gap_v5 import build_world_model
from obstacle_probe import _det_normal, _make_det, _action_dim, DEFAULT_RUNS


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _feat(wm, state) -> np.ndarray:
    """Full pooled latent feature get_feat(state) → (feat_dim,) numpy.

    get_feat = cat(deter, stoch) pooled to (B, feat_dim) for both RSSM and ViT,
    so it captures the deterministic context that actually carries dynamics —
    the right quantity for measuring whether the latent moves during rollout.
    """
    return wm.rssm.get_feat(state).squeeze(0).cpu().numpy()


def _mean_feat(state) -> np.ndarray:
    """The pooled posterior mean — the exact feature the obstacle probe reads."""
    return state.mean.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame rollout with GT actions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_one(
    wm,
    f,                       # open h5py.File
    idx: int,
    episode_ids_all: np.ndarray,
    context_len: int,
    max_h: int,
    device: torch.device,
):
    """Encode start posterior at `idx`, then roll out max_h prior steps using GT
    future actions.  Returns dicts keyed by h with feature/recon arrays, or None
    if there are not enough future frames in the same episode.
    """
    action_dim = _action_dim(wm)
    ep = int(episode_ids_all[idx])
    ep_idxs = np.where(episode_ids_all == ep)[0]
    ep_start, ep_end = int(ep_idxs[0]), int(ep_idxs[-1])
    if idx + max_h > ep_end:
        return None  # not enough future in this episode

    # ── Start posterior via short context ─────────────────────────────────
    ctx_start = max(ep_start, idx - context_len + 1)
    cs = slice(ctx_start, idx + 1)
    ctx_img = torch.tensor(f["images"][cs],  dtype=torch.float32)
    ctx_st  = torch.tensor(f["states"][cs],  dtype=torch.float32)
    ctx_act = torch.tensor(f["actions"][cs], dtype=torch.float32)

    state    = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, action_dim, device=device)
    with _det_normal():
        for t in range(len(ctx_img)):
            embed = wm.encode_obs(ctx_img[t:t+1].to(device), ctx_st[t:t+1].to(device))
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            state   = _make_det(post)
            prev_act = ctx_act[t:t+1].to(device)
    start_state = state

    # Future actions + GT future frames (preprocessed to model space)
    fs = slice(idx + 1, idx + max_h + 1)
    fut_act = torch.tensor(f["actions"][fs], dtype=torch.float32)
    fut_img = torch.tensor(f["images"][fs],  dtype=torch.float32)

    feat0      = _feat(wm, start_state)
    mean0      = _mean_feat(start_state)
    prev_feat  = feat0
    prev_mean  = mean0

    out = {
        "drift_from_start": {},
        "drift_step":       {},
        "mean_drift_start": {},
        "recon_mse":        {},
    }

    with _det_normal():
        s = start_state
        for h in range(1, max_h + 1):
            act = fut_act[h - 1:h].to(device)
            s   = _make_det(wm.rssm.img_step(s, act))

            fh   = _feat(wm, s)
            mh   = _mean_feat(s)
            out["drift_from_start"][h] = float(np.linalg.norm(fh - feat0))
            out["drift_step"][h]       = float(np.linalg.norm(fh - prev_feat))
            out["mean_drift_start"][h] = float(np.linalg.norm(mh - mean0))
            prev_feat = fh
            prev_mean = mh

            # Rollout reconstruction MSE vs GT future frame
            recon  = wm.decode_obs(s)
            target = wm.preprocess_image(fut_img[h - 1:h].to(device))
            out["recon_mse"][h] = float((recon - target).pow(2).mean().item())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate over samples
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_model(wm, path, sel_idx, episode_ids_all, context_len, max_h, device):
    keys = ["drift_from_start", "drift_step", "mean_drift_start", "recon_mse"]
    acc = {k: {h: [] for h in range(1, max_h + 1)} for k in keys}
    n_used = 0
    with h5py.File(path, "r") as f:
        for idx in sel_idx:
            r = rollout_one(wm, f, int(idx), episode_ids_all,
                            context_len, max_h, device)
            if r is None:
                continue
            n_used += 1
            for k in keys:
                for h, v in r[k].items():
                    acc[k][h].append(v)

    summary = {k: {} for k in keys}
    for k in keys:
        for h in range(1, max_h + 1):
            vals = acc[k][h]
            if vals:
                summary[k][h] = {"mean": float(np.mean(vals)),
                                 "std":  float(np.std(vals))}
    return summary, n_used


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_drift(results: dict, max_h: int, output_png: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    COLORS = {"rssm_rr_off": "#5599dd", "rssm_rr_w2.0": "#1144aa",
               "vit_rr_off": "#ee8833", "vit_rr_w2.0": "#cc2200"}

    panels = [
        ("drift_from_start", "‖feat_h − feat_0‖  (drift from start)"),
        ("drift_step",       "‖feat_h − feat_{h-1}‖  (per-step change)"),
        ("recon_mse",        "rollout decode MSE vs GT future"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=120)

    for (key, ylabel), ax in zip(panels, axes):
        for name, summ in results.items():
            d = summ.get(key, {})
            hs = sorted(d.keys())
            if not hs:
                continue
            ys  = [d[h]["mean"] for h in hs]
            err = [d[h]["std"]  for h in hs]
            col = COLORS.get(name, None)
            ax.plot(hs, ys, marker="o", markersize=4, linewidth=1.8,
                    label=name, color=col)
            ax.fill_between(hs, [y - e for y, e in zip(ys, err)],
                                 [y + e for y, e in zip(ys, err)],
                            alpha=0.12, color=col)
        ax.set_xlabel("rollout step h", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rollout latent drift + decode MSE check",
    )
    parser.add_argument("--data", default="data/expert_obstacle_v5.h5")
    parser.add_argument("--runs_json", default=None,
                        help="Model specs; default = rollout-recon variants")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max obstacle-visible start frames to roll out")
    parser.add_argument("--max_h",       type=int, default=20)
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/rollout_drift")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", type=int, default=10,
                        help="Max CPU threads for BLAS/torch (default 10)")
    args = parser.parse_args()

    n_workers = max(1, args.worker)
    torch.set_num_threads(n_workers)
    print(f"CPU thread cap: {n_workers} workers "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = DEFAULT_RUNS if not args.runs_json else json.load(open(args.runs_json))
    runs = [r for r in runs if Path(r["checkpoint"]).exists()]
    if not runs:
        print("ERROR: no valid checkpoints"); sys.exit(1)

    # Select obstacle-visible start frames
    print(f"Loading metadata from {args.data} …")
    with h5py.File(args.data, "r") as f:
        episode_ids_all = f["episode_ids"][:]
        obs_vis_all     = f["obstacle_visible"][:].astype(bool)
    vis_idx = np.where(obs_vis_all)[0]
    rng = np.random.default_rng(args.seed)
    if len(vis_idx) > args.max_samples:
        vis_idx = np.sort(rng.choice(vis_idx, args.max_samples, replace=False))
    print(f"  {len(vis_idx)} obstacle-visible start frames "
          f"(of {int(obs_vis_all.sum())} total)")

    results: dict[str, dict] = {}
    for spec in runs:
        name = spec["name"]
        print(f"\n[{name}] loading …")
        wm, _, cfg = build_world_model(
            spec["config"], spec["checkpoint"], device,
            overrides=spec.get("overrides") or None,
        )
        wm._cfg = cfg
        print(f"  rolling out {len(vis_idx)} frames × {args.max_h} steps …",
              end="", flush=True)
        summ, n_used = run_model(
            wm, args.data, vis_idx, episode_ids_all,
            args.context_len, args.max_h, device,
        )
        results[name] = summ
        print(f" done ({n_used} usable)")

        # Console summary at key horizons
        for h in [1, 5, 10, 20]:
            if h in summ["drift_from_start"]:
                ds = summ["drift_from_start"][h]["mean"]
                st = summ["drift_step"][h]["mean"]
                mse = summ["recon_mse"][h]["mean"]
                print(f"    h={h:2d}  drift_from_start={ds:.4f}  "
                      f"drift_step={st:.4f}  recon_mse={mse:.5f}")

    # Save JSON
    json_path = out_dir / "rollout_drift_results.json"
    with open(json_path, "w") as fp:
        json.dump({n: {k: {str(h): v for h, v in d.items()}
                       for k, d in summ.items()}
                   for n, summ in results.items()}, fp, indent=2)
    print(f"\n  JSON → {json_path}")

    plot_drift(results, args.max_h, str(out_dir / "rollout_drift_curve.png"))

    # Verdict hint
    print("\n" + "=" * 64)
    print("Verdict hints")
    print("=" * 64)
    for name, summ in results.items():
        if args.max_h in summ["drift_from_start"]:
            d20 = summ["drift_from_start"][args.max_h]["mean"]
            d1  = summ["drift_from_start"].get(1, {}).get("mean", float("nan"))
            mse20 = summ["recon_mse"][args.max_h]["mean"]
            print(f"  {name:14s}  drift@h{args.max_h}={d20:.4f}  "
                  f"(h1={d1:.4f})  recon_mse@h{args.max_h}={mse20:.5f}")
    print("  → near-zero drift = rollout frozen (probe = artifact).")
    print("  → higher recon_mse with higher probe = probe unsuitable.")
    print("\nDone.")


if __name__ == "__main__":
    main()

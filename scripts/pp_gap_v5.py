"""Posterior-Prior Gap evaluation for v5 data.

Measures how well the world model's transition prior retains the preference
separation between clean and obstacle states over multiple imagination horizons.

Definitions
-----------
    delta_post        = mean(score(clean_post)) - mean(score(obstacle_post))
    delta_prior(h)    = mean(score(clean_prior_h)) - mean(score(obstacle_prior_h))
    PP_Gap(h)         = delta_post - delta_prior(h)
    Retention(h)      = delta_prior(h) / delta_post  [when delta_post != 0]

Input HDF5
----------
--eval_data must be a grouped HDF5 with keys:
    eval_data["clean"]            : clean episode frames
    eval_data["obstacle_visible"] : visible-obstacle frames

Usage
-----
    uv run python scripts/pp_gap_v5.py \
        --eval_data data/obstacle_visible_eval_v5.h5 \
        --rssm_checkpoint outputs/rssm_v5/checkpoints/contrastive_pref_v5.pt \
        --tokenvit_checkpoint outputs/token_vit_v5/checkpoints/contrastive_pref_v5.pt \
        --horizons 1,3,5,7,10,12,15 \
        --output_json outputs/pp_gap_v5_results.json \
        --plot_png outputs/pp_gap_v5_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor

from active_inference.config import Config
from active_inference.agent import WorldModel


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 loading
# ─────────────────────────────────────────────────────────────────────────────

def load_group(path: str, group: str) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
    """Load images, states, actions, episode_ids from an HDF5 group."""
    with h5py.File(path, "r") as f:
        grp = f[group]
        images = torch.tensor(np.array(grp["images"]), dtype=torch.float32)
        states = torch.tensor(np.array(grp["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(grp["actions"]), dtype=torch.float32)
        episode_ids = np.array(grp["episode_ids"])
    return images, states, actions, episode_ids


# ─────────────────────────────────────────────────────────────────────────────
# GMM log-probability
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_log_prob(
    z: Tensor,
    means: Tensor,
    log_stds: Tensor,
    logits: Tensor,
    min_std: float = 0.01,
) -> Tensor:
    """Diagonal Gaussian mixture log probability.  z: (..., D)."""
    z = z.unsqueeze(-2)
    means = means.to(z.device).unsqueeze(0)
    log_stds = log_stds.to(z.device).unsqueeze(0)
    logits = logits.to(z.device)

    stds = log_stds.exp().clamp_min(min_std)
    log_w = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_g = (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(stds)
        - 0.5 * ((z - means) / stds).pow(2)
    ).sum(dim=-1)
    return torch.logsumexp(log_g + log_w, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Preference scorer
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveScorer:
    """log p_clean(z) - scale * log p_avoid(z) from checkpoint."""

    def __init__(self, ckpt: dict, device: torch.device):
        cp = ckpt["contrastive_preference"]
        self.clean_means    = cp["clean_means"].to(device)
        self.clean_log_stds = cp["clean_log_stds"].to(device)
        self.clean_logits   = cp["clean_logits"].to(device)
        self.avoid_means    = cp["avoid_means"].to(device)
        self.avoid_log_stds = cp["avoid_log_stds"].to(device)
        self.avoid_logits   = cp["avoid_logits"].to(device)
        self.scale = float(cp.get("contrast_scale", 1.0))

    def __call__(self, state: Any) -> Tensor:
        z = state.mean
        c = _gmm_log_prob(z, self.clean_means, self.clean_log_stds, self.clean_logits)
        a = _gmm_log_prob(z, self.avoid_means, self.avoid_log_stds, self.avoid_logits)
        return c - self.scale * a


# ─────────────────────────────────────────────────────────────────────────────
# Posterior encoding (same logic as updated encode_latents in
# fit_contrastive_preference.py: reset state at episode boundaries)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_posterior(
    wm: WorldModel,
    images: Tensor,
    states: Tensor,
    actions: Tensor,
    episode_ids: np.ndarray,
    device: torch.device,
    max_samples: int = 5000,
) -> list:
    """Return list of RSSMState posteriors (one per frame, up to max_samples)."""
    n = min(len(images), max_samples)
    action_dim = actions.shape[-1]
    rssm_state = wm.rssm.initial(1, device)
    prev_act = torch.zeros(1, action_dim, device=device)
    post_states = []

    for i in range(n):
        if i > 0 and episode_ids[i] != episode_ids[i - 1]:
            rssm_state = wm.rssm.initial(1, device)
            prev_act = torch.zeros(1, action_dim, device=device)
        img = images[i:i+1].to(device)
        st  = states[i:i+1].to(device)
        embed = wm.encode_obs(img, st)
        post, _ = wm.rssm.obs_step(rssm_state, prev_act, embed)
        post_states.append(type(post)(*[x.detach() for x in post]))
        rssm_state = post_states[-1]
        prev_act = actions[i:i+1].to(device)

    return post_states


# ─────────────────────────────────────────────────────────────────────────────
# Prior imagination
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def imagine_from_states(
    wm: WorldModel,
    initial_states: list,
    horizon: int,
    device: torch.device,
    action_mode: str = "zero",
    actions_ref: Tensor | None = None,
    start_indices: list[int] | None = None,
) -> list:
    """Imagine h steps forward from each initial state.

    Args:
        wm: world model
        initial_states: list of N RSSMState objects
        horizon: number of imagination steps
        action_mode: "zero" (zero actions) or "expert" (actions_ref[start+t])
        actions_ref: (T, action_dim) tensor of reference actions
        start_indices: frame indices corresponding to initial_states

    Returns list of N RSSMState objects, each h steps ahead.
    """
    action_dim = wm.rssm._img_in[0].in_features  # stoch+action_dim
    # Derive action_dim: img_in expects (stoch_dim + action_dim) input
    stoch_dim = initial_states[0].stoch.shape[-1]
    action_dim_actual = wm.rssm._img_in[0].in_features - stoch_dim

    final_states = []
    for idx, init_state in enumerate(initial_states):
        state = init_state
        for step in range(horizon):
            if action_mode == "expert" and actions_ref is not None and start_indices is not None:
                frame_idx = start_indices[idx] + step
                if frame_idx < len(actions_ref):
                    act = actions_ref[frame_idx:frame_idx+1].to(device)
                else:
                    act = torch.zeros(1, action_dim_actual, device=device)
            else:
                act = torch.zeros(1, action_dim_actual, device=device)
            state = wm.rssm.img_step(state, act)
        final_states.append(type(state)(*[x.detach() for x in state]))
    return final_states


# ─────────────────────────────────────────────────────────────────────────────
# PP-Gap computation for one model
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_pp_gap(
    wm: WorldModel,
    scorer: ContrastiveScorer,
    clean_images: Tensor,
    clean_states: Tensor,
    clean_actions: Tensor,
    clean_episode_ids: np.ndarray,
    obs_images: Tensor,
    obs_states: Tensor,
    obs_actions: Tensor,
    obs_episode_ids: np.ndarray,
    horizons: list[int],
    device: torch.device,
    max_samples: int = 2000,
    action_mode: str = "zero",
) -> dict:
    """Compute PP-Gap metrics for a single model."""
    # ── Encode posteriors ──────────────────────────────────────────────
    clean_posts = encode_posterior(
        wm, clean_images, clean_states, clean_actions, clean_episode_ids,
        device, max_samples,
    )
    obs_posts = encode_posterior(
        wm, obs_images, obs_states, obs_actions, obs_episode_ids,
        device, max_samples,
    )

    # ── Posterior scores ───────────────────────────────────────────────
    clean_scores_post = torch.stack([scorer(s).squeeze() for s in clean_posts])
    obs_scores_post   = torch.stack([scorer(s).squeeze() for s in obs_posts])

    delta_post = float(clean_scores_post.mean().item()) - float(obs_scores_post.mean().item())

    # ── Horizon loop ───────────────────────────────────────────────────
    delta_prior_list: list[float] = []
    pp_gap_list: list[float] = []
    retention_list: list[float] = []

    # Use the start_indices = consecutive frame indices for expert mode
    clean_start = list(range(len(clean_posts)))
    obs_start   = list(range(len(obs_posts)))

    for h in horizons:
        clean_prior_states = imagine_from_states(
            wm, clean_posts, h, device, action_mode,
            actions_ref=clean_actions, start_indices=clean_start,
        )
        obs_prior_states = imagine_from_states(
            wm, obs_posts, h, device, action_mode,
            actions_ref=obs_actions, start_indices=obs_start,
        )

        clean_scores_prior = torch.stack([scorer(s).squeeze() for s in clean_prior_states])
        obs_scores_prior   = torch.stack([scorer(s).squeeze() for s in obs_prior_states])

        d_prior = float(clean_scores_prior.mean().item()) - float(obs_scores_prior.mean().item())
        pp_gap = delta_post - d_prior
        retention = d_prior / delta_post if abs(delta_post) > 1e-8 else float("nan")

        delta_prior_list.append(d_prior)
        pp_gap_list.append(pp_gap)
        retention_list.append(retention)

    return {
        "delta_post": delta_post,
        "horizons": horizons,
        "delta_prior": delta_prior_list,
        "pp_gap": pp_gap_list,
        "retention": retention_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build world model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def build_world_model(cfg_path: str, ckpt_path: str, device: torch.device) -> tuple:
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"], strict=True)
    wm.eval()
    return wm, ckpt, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_retention(results: dict, output_png: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"rssm": "#1f77b4", "tokenvit": "#ff7f0e"}

    for model_name, res in results.items():
        horizons = res["horizons"]
        retention = [r if not math.isnan(r) else 0.0 for r in res["retention"]]
        c = colors.get(model_name, None)
        ax.plot(horizons, retention, marker="o", label=model_name, color=c)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0,
               label="Retention = 1 (ideal)")
    ax.set_xlabel("Imagination Horizon (steps)")
    ax.set_ylabel("Retention  Δ_prior / Δ_post")
    ax.set_title("Posterior-Prior Gap — Retention Curve")
    ax.set_ylim(-0.2, 1.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute PP-Gap for RSSM and TokenViT on v5 eval data",
    )
    parser.add_argument("--eval_data",
        default="data/obstacle_visible_eval_v5.h5",
        help="HDF5 with groups 'clean' and 'obstacle_visible'")
    parser.add_argument("--rssm_checkpoint", default=None)
    parser.add_argument("--rssm_config",
        default="configs/experiment/task_b_pure_aif.yaml")
    parser.add_argument("--tokenvit_checkpoint", default=None)
    parser.add_argument("--tokenvit_config",
        default="configs/experiment/token_vit.yaml")
    parser.add_argument("--horizons", default="1,3,5,7,10,12,15")
    parser.add_argument("--output_json",
        default="outputs/pp_gap_v5_results.json")
    parser.add_argument("--plot_png",
        default="outputs/pp_gap_v5_plot.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--imagine_action_mode",
        default="zero", choices=["zero", "expert"],
        help="zero = zero-action imagination; expert = use reference actions")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading eval data from {args.eval_data} …")
    clean_images, clean_states, clean_actions, clean_ep_ids = load_group(
        args.eval_data, "clean",
    )
    obs_images, obs_states, obs_actions, obs_ep_ids = load_group(
        args.eval_data, "obstacle_visible",
    )
    print(f"  clean: {len(clean_images)} frames, "
          f"obstacle_visible: {len(obs_images)} frames")

    results = {}

    checkpoints = []
    if args.rssm_checkpoint:
        checkpoints.append(("rssm", args.rssm_checkpoint, args.rssm_config))
    if args.tokenvit_checkpoint:
        checkpoints.append(("tokenvit", args.tokenvit_checkpoint, args.tokenvit_config))

    if not checkpoints:
        print("ERROR: at least one of --rssm_checkpoint or --tokenvit_checkpoint required")
        sys.exit(1)

    for model_name, ckpt_path, cfg_path in checkpoints:
        print(f"\n[{model_name}] Loading from {ckpt_path} …")
        wm, ckpt, cfg = build_world_model(cfg_path, ckpt_path, device)

        if "contrastive_preference" not in ckpt:
            print(f"  WARNING: no contrastive_preference in checkpoint, skipping {model_name}")
            continue

        scorer = ContrastiveScorer(ckpt, device)
        print(f"  Scorer: ContrastiveScorer")

        res = compute_pp_gap(
            wm=wm,
            scorer=scorer,
            clean_images=clean_images,
            clean_states=clean_states,
            clean_actions=clean_actions,
            clean_episode_ids=clean_ep_ids,
            obs_images=obs_images,
            obs_states=obs_states,
            obs_actions=obs_actions,
            obs_episode_ids=obs_ep_ids,
            horizons=horizons,
            device=device,
            max_samples=args.max_samples,
            action_mode=args.imagine_action_mode,
        )
        results[model_name] = res

        print(f"  delta_post = {res['delta_post']:+.4f}")
        for h, dp, gap, ret in zip(
            res["horizons"], res["delta_prior"], res["pp_gap"], res["retention"]
        ):
            print(f"  h={h:2d}: delta_prior={dp:+.4f}  PP_Gap={gap:+.4f}  "
                  f"Retention={ret:.4f}" if not math.isnan(ret) else
                  f"  h={h:2d}: delta_prior={dp:+.4f}  PP_Gap={gap:+.4f}  "
                  f"Retention=NaN")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved → {args.output_json}")

    if results:
        plot_retention(results, args.plot_png)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Public API (used by tests)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "encode_posterior",
    "imagine_from_states",
    "compute_pp_gap",
    "ContrastiveScorer",
    "_gmm_log_prob",
]

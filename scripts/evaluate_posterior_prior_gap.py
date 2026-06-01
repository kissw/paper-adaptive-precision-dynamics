#!/usr/bin/env python
"""Evaluate Posterior-Prior Gap using preference scores.

This script directly measures whether task-relevant posterior information
survives action-conditioned prior imagination.

Definitions
-----------
For a scalar preference score s(z), where higher = more clean/preferred:

    posterior score:
        s_post(t+h)  = s(q(z_t+h | GT future image))

    prior score:
        s_prior(t+h) = s(p(z_t+h | context, actions))

For clean and obstacle datasets:

    �_post  = E_clean[s_post]  - E_obstacle[s_post]
    �_prior = E_clean[s_prior] - E_obstacle[s_prior]
    PP-Gap  = �_post - �_prior
    Retention = �_prior / �_post

Interpretation
--------------
�_post large:
    Posterior representation separates clean vs obstacle.

�_prior close to �_post:
    Transition prior preserves that separation during imagination.

PP-Gap large:
    Posterior knows the difference, but prior imagination loses it.

This is decoder-free. No image decoder, PSNR, or SSIM is used.

Supported preference checkpoints
--------------------------------
1. pooled_contrastive:
   checkpoint["contrastive_preference"]
   score(state) = log p_clean(state.mean) - scale * log p_avoid(state.mean)

2. pooled_single:
   checkpoint["preference"]
   score(state) = log p_clean(state.mean)

3. token_contrastive:
   checkpoint["token_contrastive_preference"]
   score(state) = top-k mean over token log-ratio scores

4. token_vampprior:
   checkpoint["token_vampprior_preference"]
   score(state) = top-k mean over position-conditional posterior mixture scores

Expected usage
--------------
RSSM pooled contrastive:

    uv run python scripts/evaluate_posterior_prior_gap.py \
        --checkpoint eval_dir/rssm_pooled_contrastive_k5k7_s20000.pt \
        --config configs/experiment/task_b_v5.yaml \
        --preference_type pooled_contrastive \
        --clean_data data/expert_data_town04.h5 \
        --obstacle_data data/expert_data_v4.h5 \
        --start_index 1000 \
        --context_len 5 \
        --horizon 15 \
        --num_cases 20 \
        --stride 500 \
        --output_dir pp_gap_rssm_pooled

TokenViT token contrastive:

    uv run python scripts/evaluate_posterior_prior_gap.py \
        --checkpoint eval_dir/tokenvit_posnorm_token_contrastive_k5k7_f20000_t200k.pt \
        --config configs/experiment/token_vit.yaml \
        --preference_type token_contrastive \
        --clean_data data/expert_data_town04.h5 \
        --obstacle_data data/expert_data_v4.h5 \
        --start_index 1000 \
        --context_len 5 \
        --horizon 15 \
        --num_cases 20 \
        --stride 500 \
        --output_dir pp_gap_tokenvit_posnorm
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from active_inference.config import Config
from active_inference.agent import WorldModel
from active_inference.training.token_preference import extract_token_features
from active_inference.training.token_vampprior_preference import (
    extract_token_posterior_stats,
    log_prob_posterior_under_mixture,
    log_prob_mean_only,
    topk_mean_score,
)


# ---------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------

def load_hdf5(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as f:
        img_key = "images" if "images" in f else "frames"
        if img_key not in f:
            raise KeyError(f"No images/frames key in {path}")
        if "states" not in f:
            raise KeyError(f"No states key in {path}")
        if "actions" not in f:
            raise KeyError(f"No actions key in {path}")

        images = torch.tensor(np.array(f[img_key]), dtype=torch.float32)
        states = torch.tensor(np.array(f["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(f["actions"]), dtype=torch.float32)

    # Some datasets store actions as T-1. Pad for safe frame indexing.
    if actions.shape[0] < images.shape[0]:
        pad = torch.zeros(images.shape[0] - actions.shape[0], actions.shape[-1])
        actions = torch.cat([actions, pad], dim=0)

    return images, states, actions


def build_world_model(cfg_path: str, ckpt_path: str, device: torch.device) -> tuple[WorldModel, dict]:
    cfg = Config.from_yaml(cfg_path)
    wm = WorldModel(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "world_model" not in ckpt:
        raise KeyError(f"{ckpt_path} has no top-level key 'world_model'")

    wm.load_state_dict(ckpt["world_model"], strict=True)
    wm.eval()
    return wm, ckpt


# ---------------------------------------------------------------------
# GMM log-prob
# ---------------------------------------------------------------------

def gmm_log_prob(
    z: Tensor,
    means: Tensor,
    log_stds: Tensor,
    logits: Tensor,
    min_std: float = 0.01,
) -> Tensor:
    """Diagonal Gaussian mixture log probability.

    Args:
        z:       (..., D)
        means:   (K, D)
        log_stds:(K, D)
        logits:  (K,)

    Returns:
        (...,)
    """
    z = z.unsqueeze(-2)                    # (..., 1, D)
    means = means.to(z.device).unsqueeze(0)
    log_stds = log_stds.to(z.device).unsqueeze(0)
    logits = logits.to(z.device)

    stds = log_stds.exp().clamp_min(min_std)
    log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    log_gauss = (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(stds)
        - 0.5 * ((z - means) / stds).pow(2)
    ).sum(dim=-1)                          # (..., K)

    return torch.logsumexp(log_gauss + log_weights, dim=-1)


# ---------------------------------------------------------------------
# Preference scorers
# ---------------------------------------------------------------------

class PreferenceScorer:
    def __call__(self, state: Any) -> Tensor:
        raise NotImplementedError


class PooledSingleScorer(PreferenceScorer):
    def __init__(self, pref: dict, device: torch.device):
        self.means = pref["means"].to(device)
        self.log_stds = pref["log_stds"].to(device)
        self.logits = pref["logits"].to(device)
        self.min_std = float(pref.get("min_std", 0.01))

    def __call__(self, state: Any) -> Tensor:
        return gmm_log_prob(
            state.mean,
            self.means,
            self.log_stds,
            self.logits,
            min_std=self.min_std,
        )


class PooledContrastiveScorer(PreferenceScorer):
    def __init__(self, pref: dict, device: torch.device):
        self.clean_means = pref["clean_means"].to(device)
        self.clean_log_stds = pref["clean_log_stds"].to(device)
        self.clean_logits = pref["clean_logits"].to(device)
        self.avoid_means = pref["avoid_means"].to(device)
        self.avoid_log_stds = pref["avoid_log_stds"].to(device)
        self.avoid_logits = pref["avoid_logits"].to(device)
        self.contrast_scale = float(pref.get("contrast_scale", 1.0))
        self.min_std = float(pref.get("min_std", 0.01))

    def __call__(self, state: Any) -> Tensor:
        z = state.mean
        clean = gmm_log_prob(
            z,
            self.clean_means,
            self.clean_log_stds,
            self.clean_logits,
            min_std=self.min_std,
        )
        avoid = gmm_log_prob(
            z,
            self.avoid_means,
            self.avoid_log_stds,
            self.avoid_logits,
            min_std=self.min_std,
        )
        return clean - self.contrast_scale * avoid


class TokenContrastiveScorer(PreferenceScorer):
    def __init__(self, pref: dict, device: torch.device, topk: int | None = None):
        self.pref = pref
        self.device = device

        self.clean_means = pref["clean_means"].to(device)
        self.clean_log_stds = pref["clean_log_stds"].to(device)
        self.clean_logits = pref["clean_logits"].to(device)
        self.avoid_means = pref["avoid_means"].to(device)
        self.avoid_log_stds = pref["avoid_log_stds"].to(device)
        self.avoid_logits = pref["avoid_logits"].to(device)

        self.contrast_scale = float(pref.get("contrast_scale", 1.0))
        self.min_std = float(pref.get("min_std", 0.01))
        self.feature_type = str(pref.get("feature_type", "deter_stoch"))
        self.topk = int(topk if topk is not None else pref.get("topk_default", 4))

        self.position_stats = pref.get("position_stats", None)
        if self.position_stats is not None:
            self.mu_pos = self.position_stats["mu_pos"].to(device)
            self.std_pos = self.position_stats["std_pos"].to(device)
        else:
            self.mu_pos = None
            self.std_pos = None

    def _tokens(self, state: Any) -> Tensor:
        z = extract_token_features(state, feature_type=self.feature_type).to(self.device)
        if self.mu_pos is not None:
            z = (z - self.mu_pos.unsqueeze(0)) / self.std_pos.unsqueeze(0)
        return z

    def __call__(self, state: Any) -> Tensor:
        z = self._tokens(state)            # (B, N, D)
        B, N, D = z.shape
        flat = z.reshape(B * N, D)

        clean = gmm_log_prob(
            flat,
            self.clean_means,
            self.clean_log_stds,
            self.clean_logits,
            min_std=self.min_std,
        )
        avoid = gmm_log_prob(
            flat,
            self.avoid_means,
            self.avoid_log_stds,
            self.avoid_logits,
            min_std=self.min_std,
        )
        token_scores = (clean - self.contrast_scale * avoid).reshape(B, N)
        k = min(self.topk, N)
        return torch.topk(token_scores, k=k, dim=-1).values.mean(dim=-1)


class TokenVampPriorScorer(PreferenceScorer):
    def __init__(self, pref: dict, device: torch.device, topk: int | None = None):
        self.device = device

        self.clean_mean = pref["clean_mean"].to(device)
        self.clean_log_std = pref["clean_log_std"].to(device)
        self.clean_logits = pref["clean_logits"].to(device)

        self.avoid_mean = pref["avoid_mean"].to(device)
        self.avoid_log_std = pref["avoid_log_std"].to(device)
        self.avoid_logits = pref["avoid_logits"].to(device)

        self.contrast_scale = float(pref.get("contrast_scale", 1.0))
        self.score_mode = str(pref.get("score_mode", "q_integrated"))
        self.q_std_scale = float(pref.get("q_std_scale", 1.0))
        self.proto_std_scale = float(pref.get("proto_std_scale", 1.0))
        self.topk = int(topk if topk is not None else pref.get("topk_default", 4))

    def __call__(self, state: Any) -> Tensor:
        q_mean, q_std = extract_token_posterior_stats(state)
        q_mean = q_mean.to(self.device)
        q_std = q_std.to(self.device)

        if self.score_mode == "mean_only":
            clean_lp = log_prob_mean_only(
                q_mean,
                self.clean_mean,
                self.clean_log_std,
                self.clean_logits,
                proto_std_scale=self.proto_std_scale,
            )
            avoid_lp = log_prob_mean_only(
                q_mean,
                self.avoid_mean,
                self.avoid_log_std,
                self.avoid_logits,
                proto_std_scale=self.proto_std_scale,
            )
        else:
            clean_lp = log_prob_posterior_under_mixture(
                q_mean,
                q_std,
                self.clean_mean,
                self.clean_log_std,
                self.clean_logits,
                q_std_scale=self.q_std_scale,
                proto_std_scale=self.proto_std_scale,
            )
            avoid_lp = log_prob_posterior_under_mixture(
                q_mean,
                q_std,
                self.avoid_mean,
                self.avoid_log_std,
                self.avoid_logits,
                q_std_scale=self.q_std_scale,
                proto_std_scale=self.proto_std_scale,
            )

        token_scores = clean_lp - self.contrast_scale * avoid_lp
        return topk_mean_score(token_scores, k=self.topk)


def build_scorer(
    ckpt: dict,
    preference_type: str,
    device: torch.device,
    topk: int | None,
) -> PreferenceScorer:
    if preference_type == "auto":
        if "token_vampprior_preference" in ckpt:
            preference_type = "token_vampprior"
        elif "token_contrastive_preference" in ckpt:
            preference_type = "token_contrastive"
        elif "contrastive_preference" in ckpt:
            preference_type = "pooled_contrastive"
        elif "preference" in ckpt:
            preference_type = "pooled_single"
        else:
            raise KeyError("No supported preference key found in checkpoint.")

    if preference_type == "pooled_single":
        return PooledSingleScorer(ckpt["preference"], device)
    if preference_type == "pooled_contrastive":
        return PooledContrastiveScorer(ckpt["contrastive_preference"], device)
    if preference_type == "token_contrastive":
        return TokenContrastiveScorer(ckpt["token_contrastive_preference"], device, topk)
    if preference_type == "token_vampprior":
        return TokenVampPriorScorer(ckpt["token_vampprior_preference"], device, topk)

    raise ValueError(f"Unknown preference_type: {preference_type}")


# ---------------------------------------------------------------------
# RSSM state helpers
# ---------------------------------------------------------------------

def detach_state(state: Any) -> Any:
    return type(state)(*[x.detach() for x in state])


def use_mean_stoch(state: Any) -> Any:
    """Use mean/token_mean instead of sampled stochastic state."""
    if hasattr(state, "token_mean"):
        return state._replace(stoch=state.token_mean)
    return state._replace(stoch=state.mean)


def get_action(actions: Tensor, idx: int, device: torch.device) -> Tensor:
    if idx < 0 or idx >= actions.shape[0]:
        return torch.zeros(1, actions.shape[-1], device=device)
    return actions[idx : idx + 1].to(device)


@torch.no_grad()
def build_context_state(
    wm: WorldModel,
    images: Tensor,
    states: Tensor,
    actions: Tensor,
    start: int,
    context_len: int,
    device: torch.device,
    deterministic: bool,
) -> Any:
    state = wm.rssm.initial(1, device)
    prev_action = torch.zeros(1, actions.shape[-1], device=device)

    for i in range(context_len):
        t = start + i
        img_t = images[t : t + 1].to(device)
        st_t = states[t : t + 1].to(device)

        embed = wm.encode_obs(img_t, st_t)
        post, _ = wm.rssm.obs_step(state, prev_action, embed)

        if deterministic:
            post = use_mean_stoch(post)

        state = detach_state(post)
        prev_action = get_action(actions, t, device)

    return state


@torch.no_grad()
def collect_scores_for_dataset(
    dataset_name: str,
    wm: WorldModel,
    scorer: PreferenceScorer,
    data_path: str,
    start_index: int,
    context_len: int,
    horizon: int,
    num_cases: int,
    stride: int,
    device: torch.device,
    deterministic: bool,
) -> list[dict[str, Any]]:
    images, states, actions = load_hdf5(data_path)
    T = images.shape[0]

    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print("path:", data_path)
    print("images:", tuple(images.shape), "range:", float(images.min()), float(images.max()))
    print("states:", tuple(states.shape))
    print("actions:", tuple(actions.shape))

    rows: list[dict[str, Any]] = []

    for case_idx in range(num_cases):
        start = start_index + case_idx * stride
        end_context = start + context_len
        end_rollout = end_context + horizon

        if end_rollout > T:
            print(
                f"[{dataset_name}] skip case={case_idx}: "
                f"start={start}, end={end_rollout}, T={T}"
            )
            continue

        t0 = time.time()

        context_state = build_context_state(
            wm=wm,
            images=images,
            states=states,
            actions=actions,
            start=start,
            context_len=context_len,
            device=device,
            deterministic=deterministic,
        )

        prior_state = context_state
        posterior_state = context_state

        for h in range(1, horizon + 1):
            frame_idx = end_context + h - 1
            action_idx = frame_idx - 1

            act_h = get_action(actions, action_idx, device)

            # Open-loop prior: no future image.
            prior = wm.rssm.img_step(prior_state, act_h)
            if deterministic:
                prior = use_mean_stoch(prior)
            prior = detach_state(prior)

            # Teacher-forced posterior target: use GT future image.
            img_h = images[frame_idx : frame_idx + 1].to(device)
            st_h = states[frame_idx : frame_idx + 1].to(device)
            embed_h = wm.encode_obs(img_h, st_h)
            post, _ = wm.rssm.obs_step(posterior_state, act_h, embed_h)
            if deterministic:
                post = use_mean_stoch(post)
            post = detach_state(post)

            s_prior = float(scorer(prior).mean().item())
            s_post = float(scorer(post).mean().item())

            rows.append({
                "dataset": dataset_name,
                "case_idx": case_idx,
                "start_index": start,
                "context_len": context_len,
                "horizon_step": h,
                "frame_index": frame_idx,
                "posterior_score": s_post,
                "prior_score": s_prior,
                "score_drop_post_minus_prior": s_post - s_prior,
            })

            prior_state = prior
            posterior_state = post

        elapsed = time.time() - t0
        print(f"[{dataset_name}] case={case_idx} start={start} done in {elapsed:.2f}s")

    return rows


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print("No rows to write:", path)
        return

    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("CSV saved ->", path)


def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute �_post, �_prior, PP-Gap by horizon and overall."""
    if not rows:
        return []

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["horizon_step"])].append(r)

    summary_rows: list[dict[str, Any]] = []

    def summarize_group(horizon_step: int, rs: list[dict[str, Any]]) -> dict[str, Any]:
        clean = [r for r in rs if r["dataset"] == "clean"]
        obst = [r for r in rs if r["dataset"] == "obstacle"]

        clean_post = mean([float(r["posterior_score"]) for r in clean])
        clean_prior = mean([float(r["prior_score"]) for r in clean])
        obst_post = mean([float(r["posterior_score"]) for r in obst])
        obst_prior = mean([float(r["prior_score"]) for r in obst])

        delta_post = clean_post - obst_post
        delta_prior = clean_prior - obst_prior
        pp_gap = delta_post - delta_prior

        retention = (
            delta_prior / delta_post
            if math.isfinite(delta_post) and abs(delta_post) > 1e-8
            else float("nan")
        )

        return {
            "horizon_step": horizon_step,
            "n_clean": len(clean),
            "n_obstacle": len(obst),

            "clean_posterior_score": clean_post,
            "clean_prior_score": clean_prior,
            "obstacle_posterior_score": obst_post,
            "obstacle_prior_score": obst_prior,

            "delta_post": delta_post,
            "delta_prior": delta_prior,
            "posterior_prior_gap": pp_gap,
            "prior_retention_ratio": retention,

            "clean_score_drop_post_minus_prior": clean_post - clean_prior,
            "obstacle_score_drop_post_minus_prior": obst_post - obst_prior,
        }

    for h in sorted(grouped.keys()):
        summary_rows.append(summarize_group(h, grouped[h]))

    summary_rows.append(summarize_group(-1, rows))
    return summary_rows


def write_summary_txt(summary_rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w") as f:
        f.write("Posterior-Prior Gap Summary\n")
        f.write("=" * 100 + "\n")
        f.write("Score convention: higher = cleaner / more preferred.\n\n")
        f.write("�_post  = E_clean[s_post]  - E_obstacle[s_post]\n")
        f.write("�_prior = E_clean[s_prior] - E_obstacle[s_prior]\n")
        f.write("PP-Gap  = �_post - �_prior\n")
        f.write("Retention = �_prior / �_post\n\n")
        f.write("Interpretation:\n")
        f.write("  �_post large  : posterior separates clean vs obstacle.\n")
        f.write("  �_prior large : prior imagination preserves that separation.\n")
        f.write("  PP-Gap large  : posterior information is lost in transition rollout.\n")
        f.write("  Retention near 1 is good; near 0 means prior lost separation.\n\n")

        for r in summary_rows:
            h = "overall" if int(r["horizon_step"]) == -1 else f"h={r['horizon_step']}"
            f.write("-" * 100 + "\n")
            f.write(f"{h}\n")
            for k in [
                "n_clean",
                "n_obstacle",
                "clean_posterior_score",
                "clean_prior_score",
                "obstacle_posterior_score",
                "obstacle_prior_score",
                "delta_post",
                "delta_prior",
                "posterior_prior_gap",
                "prior_retention_ratio",
                "clean_score_drop_post_minus_prior",
                "obstacle_score_drop_post_minus_prior",
            ]:
                v = r.get(k)
                if isinstance(v, float):
                    f.write(f"  {k:36s}: {v:+.6f}\n")
                else:
                    f.write(f"  {k:36s}: {v}\n")
            f.write("\n")

    print("TXT saved ->", path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute preference-score Posterior-Prior Gap.",
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--preference_type",
        default="auto",
        choices=[
            "auto",
            "pooled_single",
            "pooled_contrastive",
            "token_contrastive",
            "token_vampprior",
        ],
    )

    parser.add_argument("--clean_data", required=True)
    parser.add_argument("--obstacle_data", required=True)

    parser.add_argument("--start_index", type=int, default=1000)
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num_cases", type=int, default=20)
    parser.add_argument("--stride", type=int, default=500)

    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--sample_stochastic_state",
        action="store_true",
        default=False,
        help="Use rsampled stochastic states. Default replaces stoch by mean/token_mean.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    deterministic = not args.sample_stochastic_state

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Posterior-Prior Gap Evaluation")
    print("=" * 100)
    print("checkpoint:", args.checkpoint)
    print("config:", args.config)
    print("preference_type:", args.preference_type)
    print("clean_data:", args.clean_data)
    print("obstacle_data:", args.obstacle_data)
    print("device:", device)
    print("deterministic mean-state:", deterministic)
    print("output_dir:", output_dir)

    wm, ckpt = build_world_model(args.config, args.checkpoint, device)
    scorer = build_scorer(ckpt, args.preference_type, device, args.topk)

    print("world_model_type:", getattr(wm, "_wm_type", None))
    print("crop_road:", getattr(wm, "_crop_road", None))
    print("scorer:", type(scorer).__name__)

    all_rows: list[dict[str, Any]] = []

    clean_rows = collect_scores_for_dataset(
        dataset_name="clean",
        wm=wm,
        scorer=scorer,
        data_path=args.clean_data,
        start_index=args.start_index,
        context_len=args.context_len,
        horizon=args.horizon,
        num_cases=args.num_cases,
        stride=args.stride,
        device=device,
        deterministic=deterministic,
    )
    all_rows.extend(clean_rows)

    obstacle_rows = collect_scores_for_dataset(
        dataset_name="obstacle",
        wm=wm,
        scorer=scorer,
        data_path=args.obstacle_data,
        start_index=args.start_index,
        context_len=args.context_len,
        horizon=args.horizon,
        num_cases=args.num_cases,
        stride=args.stride,
        device=device,
        deterministic=deterministic,
    )
    all_rows.extend(obstacle_rows)

    summary_rows = make_summary(all_rows)

    write_csv(all_rows, output_dir / "posterior_prior_gap_scores.csv")
    write_csv(summary_rows, output_dir / "posterior_prior_gap_summary.csv")
    write_summary_txt(summary_rows, output_dir / "posterior_prior_gap_summary.txt")

    print("\nDone.")
    print("scores :", output_dir / "posterior_prior_gap_scores.csv")
    print("summary:", output_dir / "posterior_prior_gap_summary.csv")
    print("txt    :", output_dir / "posterior_prior_gap_summary.txt")


if __name__ == "__main__":
    main()
"""Fit a contrastive preference model (log-ratio of two GMMs).

Uses an existing world model checkpoint to encode latents from:
1. Obstacle-free expert data → p_clean(z)
2. Obstacle-present data → p_avoid(z)

The contrastive preference log p_clean(z) - log p_avoid(z) produces
positive scores for obstacle-free states and negative for obstacle states,
enabling EFE-driven obstacle avoidance.

Usage:
    uv run python scripts/fit_contrastive_preference.py \
        --checkpoint outputs/train_v6_obstacle_aware/checkpoints/best.pt \
        --clean_data data/expert_data_mixed.h5 \
        --obstacle_data data/expert_obstacle_avoidance.h5 \
        --output outputs/train_v6_obstacle_aware/checkpoints/contrastive_pref.pt
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.models.rssm import RSSMState
from active_inference.training.preference import ContrastivePreferenceModel
from active_inference.training.token_preference import (
    TokenContrastivePreference,
    extract_token_features,
)


def encode_latents(agent, data_path, max_samples=5000):
    """Encode HDF5 data into RSSM posterior means.

    Improvements over the original version:
    - Loads actions and episode_ids from the HDF5 file.
    - Uses prev_act = actions[i-1] at each step (correct temporal context).
    - Resets RSSM state and prev_act at episode boundaries so that
      cross-episode state bleed-over is eliminated.
    """
    with h5py.File(data_path, "r") as f:
        n = min(len(f["images"]), max_samples)
        images = torch.tensor(f["images"][:n], dtype=torch.float32)
        states = torch.tensor(f["states"][:n], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:n], dtype=torch.float32)
        episode_ids = f["episode_ids"][:n]

    wm = agent.world_model
    dev = agent._device
    latents = []
    action_dim = agent._cfg.cem.action_dim

    with torch.no_grad():
        state = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, action_dim, device=dev)
        for i in range(n):
            if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                state = wm.rssm.initial(1, dev)
                prev_act = torch.zeros(1, action_dim, device=dev)
            img = images[i : i + 1].to(dev)
            st = states[i : i + 1].to(dev)
            embed = wm.encode_obs(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            latents.append(post.mean.cpu())
            state = type(post)(*[x.detach() for x in post])
            prev_act = actions[i : i + 1].to(dev)

    return torch.cat(latents).to(dev)


def encode_token_latents(agent, data_path, max_samples=5000):
    """Encode HDF5 data into per-token features (M, token_dim) for TokenViT.

    Returns a (M, token_dim) tensor where M = n_frames * N_tokens,
    suitable for fitting TokenContrastivePreference.
    """
    with h5py.File(data_path, "r") as f:
        n = min(len(f["images"]), max_samples)
        images = torch.tensor(f["images"][:n], dtype=torch.float32)
        states = torch.tensor(f["states"][:n], dtype=torch.float32)
        actions = torch.tensor(f["actions"][:n], dtype=torch.float32)
        episode_ids = f["episode_ids"][:n]

    wm = agent.world_model
    dev = agent._device
    action_dim = agent._cfg.cem.action_dim
    all_token_feats = []

    with torch.no_grad():
        state = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, action_dim, device=dev)
        for i in range(n):
            if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                state = wm.rssm.initial(1, dev)
                prev_act = torch.zeros(1, action_dim, device=dev)
            img = images[i : i + 1].to(dev)
            st = states[i : i + 1].to(dev)
            embed = wm.encode_obs(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            z_tokens = extract_token_features(post, "deter_stoch")  # (1, N, D+Z)
            all_token_feats.append(z_tokens.squeeze(0).cpu())       # (N, D+Z)
            state = type(post)(*[x.detach() for x in post])
            prev_act = actions[i : i + 1].to(dev)

    return torch.cat(all_token_feats, dim=0)  # (n*N, token_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Fit contrastive preference model",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--clean_data", required=True,
        help="HDF5 with obstacle-free expert data",
    )
    parser.add_argument(
        "--obstacle_data", required=True,
        help="HDF5 with obstacle-present data",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/experiment/task_b_pure_aif.yaml",
    )
    parser.add_argument("--K_clean", type=int, default=5)
    parser.add_argument("--K_avoid", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--contrast_scale", type=float, default=1.0)
    parser.add_argument(
        "--token_wise", action="store_true", default=False,
        help="A2: Fit TokenContrastivePreference on per-token features (TokenViT only).",
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides, 예: token_vit.use_action_warp=true",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config, overrides=args.set or None)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    if args.token_wise:
        # A2: Token-wise contrastive preference
        wm_type = getattr(getattr(cfg, "model", None), "world_model_type", "rssm")
        if wm_type != "token_vit":
            raise ValueError("--token_wise requires model.world_model_type=token_vit")
        tv = cfg.token_vit
        token_dim = tv.deter_dim + tv.stoch_dim  # deter_stoch feature

        print(f"Encoding clean token features from {args.clean_data}...")
        clean_tokens = encode_token_latents(agent, args.clean_data, args.max_samples)
        print(f"  {clean_tokens.shape[0]} token samples ({clean_tokens.shape[0]//tv.num_tokens} frames × {tv.num_tokens} tokens)")

        print(f"Encoding obstacle token features from {args.obstacle_data}...")
        avoid_tokens = encode_token_latents(agent, args.obstacle_data, args.max_samples)
        print(f"  {avoid_tokens.shape[0]} token samples")

        contrastive_token = TokenContrastivePreference(
            K_clean=args.K_clean,
            K_avoid=args.K_avoid,
            token_dim=token_dim,
            contrast_scale=args.contrast_scale,
        ).to(agent._device)
        clean_tokens = clean_tokens.to(agent._device)
        avoid_tokens = avoid_tokens.to(agent._device)
        contrastive_token.fit(clean_tokens, avoid_tokens)

        ckpt = torch.load(args.checkpoint, map_location=agent._device, weights_only=False)
        ckpt["token_contrastive_preference"] = {
            "clean_means": contrastive_token.clean.means.data,
            "clean_log_stds": contrastive_token.clean.log_stds.data,
            "clean_logits": contrastive_token.clean.logits.data,
            "avoid_means": contrastive_token.avoid.means.data,
            "avoid_log_stds": contrastive_token.avoid.log_stds.data,
            "avoid_logits": contrastive_token.avoid.logits.data,
            "contrast_scale": args.contrast_scale,
            "K_clean": args.K_clean,
            "K_avoid": args.K_avoid,
            "token_dim": token_dim,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.output)
        print(f"\nSaved token contrastive preference to {args.output}")
        return

    print(f"Encoding clean latents from {args.clean_data}...")
    clean_latents = encode_latents(agent, args.clean_data, args.max_samples)
    print(f"  {clean_latents.shape[0]} latents encoded")

    print(f"Encoding obstacle latents from {args.obstacle_data}...")
    avoid_latents = encode_latents(
        agent, args.obstacle_data, args.max_samples,
    )
    print(f"  {avoid_latents.shape[0]} latents encoded")

    # Create and fit contrastive model
    contrastive = ContrastivePreferenceModel(
        K_clean=args.K_clean,
        K_avoid=args.K_avoid,
        latent_dim=cfg.rssm.stoch_dim,
        contrast_scale=args.contrast_scale,
    ).to(agent._device)

    contrastive.fit(clean_latents, avoid_latents)

    # Verify the log-ratio works on held-out samples
    with torch.no_grad():
        clean_score = contrastive.log_prob(clean_latents).mean().item()
        avoid_score = contrastive.log_prob(avoid_latents).mean().item()
        print(f"\nFinal verification:")
        print(f"  Clean latents avg log-ratio:    {clean_score:+.2f}")
        print(f"  Obstacle latents avg log-ratio: {avoid_score:+.2f}")
        print(f"  Gap: {clean_score - avoid_score:.2f} nats")

    # Save: pack both GMMs + world model into checkpoint
    ckpt = torch.load(args.checkpoint, map_location=agent._device, weights_only=False)
    ckpt["contrastive_preference"] = {
        "clean_means": contrastive.clean.means.data,
        "clean_log_stds": contrastive.clean.log_stds.data,
        "clean_logits": contrastive.clean.logits.data,
        "avoid_means": contrastive.avoid.means.data,
        "avoid_log_stds": contrastive.avoid.log_stds.data,
        "avoid_logits": contrastive.avoid.logits.data,
        "contrast_scale": args.contrast_scale,
        "K_clean": args.K_clean,
        "K_avoid": args.K_avoid,
    }
    # Also save standard preference (clean GMM) for backward compat
    ckpt["preference"] = {
        "means": contrastive.clean.means.data,
        "log_stds": contrastive.clean.log_stds.data,
        "logits": contrastive.clean.logits.data,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"\nSaved contrastive preference to {args.output}")


if __name__ == "__main__":
    main()

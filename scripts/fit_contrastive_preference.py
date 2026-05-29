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
import torch
from torch.utils.data import DataLoader

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.models.rssm import RSSMState
from active_inference.training.preference import ContrastivePreferenceModel


def encode_latents(agent, data_path, max_samples=5000):
    """Encode HDF5 data into RSSM posterior means."""
    with h5py.File(data_path, "r") as f:
        images = torch.tensor(f["images"][:max_samples], dtype=torch.float32)
        states = torch.tensor(f["states"][:max_samples], dtype=torch.float32)

    wm = agent.world_model
    dev = agent._device
    latents = []

    with torch.no_grad():
        state = wm.rssm.initial(1, dev)
        prev_act = torch.zeros(1, agent._cfg.cem.action_dim, device=dev)
        for i in range(min(len(images), max_samples)):
            img = images[i : i + 1].to(dev)
            st = states[i : i + 1].to(dev)
            embed = wm.encode_obs(img, st)
            post, _ = wm.rssm.obs_step(state, prev_act, embed)
            latents.append(post.mean.cpu())
            state = type(post)(*[x.detach() for x in post])

    return torch.cat(latents).to(dev)


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
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

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

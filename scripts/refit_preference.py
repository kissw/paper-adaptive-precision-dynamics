"""Re-fit the GMM preference model using only high-speed (driving) frames.

The original GMM included standing-still frames, causing the agent to
prefer braking. This script encodes preference data through the world
model, filters to only moving frames (speed > threshold), and re-fits
the GMM. The world model weights are NOT changed.

Usage:
    uv run python scripts/refit_preference.py \
        --checkpoint outputs/train_v1/checkpoints/final.pt \
        --data data/expert_data.h5 \
        --output outputs/train_v1/checkpoints/final_refit.pt \
        --min_speed 1.0
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.models.rssm import RSSMState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_speed", type=float, default=1.0,
                        help="Minimum speed (m/s) for preference frames")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--fit_iters", type=int, default=300)
    parser.add_argument("--fit_lr", type=float, default=0.01)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    device = agent._device
    wm = agent.world_model
    state_dim = cfg.encoder.state_dim

    # Load raw data
    with h5py.File(args.data, "r") as f:
        images = f["images"][:]
        states = f["states"][:]
        actions = f["actions"][:]
        episode_ids = f["episode_ids"][:]
        # Use success_flags if available
        if "success_flags" in f:
            success_flags = f["success_flags"][:]
        else:
            success_flags = np.ones(len(images), dtype=bool)

    print(f"Loaded {len(images)} frames")
    print(f"States shape: {states.shape}")

    # Slice state to match model's expected dim
    if states.shape[1] > state_dim:
        states = states[:, :state_dim]

    # Filter: only successful frames with speed > min_speed
    speeds = states[:, 0]  # speed is first state dimension
    mask = success_flags & (speeds > args.min_speed)
    valid_indices = np.where(mask)[0]
    print(f"Frames with speed > {args.min_speed} m/s and success: {len(valid_indices)}/{len(images)}")

    if len(valid_indices) < 100:
        print("ERROR: Too few valid frames for GMM fitting")
        return

    # Encode filtered frames through world model to get latent states
    latents = []
    seq_len = cfg.training.seq_len
    batch_size = 16

    # Process by episodes to maintain temporal coherence
    unique_episodes = np.unique(episode_ids[valid_indices])
    print(f"Processing {len(unique_episodes)} episodes...")

    with torch.no_grad():
        count = 0
        for ep_id in unique_episodes:
            if count >= args.max_samples:
                break

            ep_mask = episode_ids == ep_id
            ep_indices = np.where(ep_mask)[0]

            if len(ep_indices) < 2:
                continue

            # Process this episode sequentially
            state = wm.rssm.initial(1, device)
            prev_act = torch.zeros(1, cfg.cem.action_dim, device=device)

            for idx in ep_indices:
                img_t = torch.tensor(images[idx], dtype=torch.float32).unsqueeze(0).to(device)
                st_t = torch.tensor(states[idx], dtype=torch.float32).unsqueeze(0).to(device)
                embed = wm.encoder(img_t, st_t)
                post, _ = wm.rssm.obs_step(state, prev_act, embed)

                # Only keep this latent if it passes the speed filter
                if mask[idx]:
                    latents.append(post.mean.cpu())
                    count += 1
                    if count >= args.max_samples:
                        break

                state = RSSMState(*[x.detach() for x in post])
                prev_act = torch.tensor(actions[idx], dtype=torch.float32).unsqueeze(0).to(device)

            if count % 500 == 0:
                print(f"  Encoded {count}/{args.max_samples} latents...")

    latent_tensor = torch.cat(latents, dim=0).to(device)
    print(f"\nTotal latents for GMM fitting: {latent_tensor.shape[0]}")

    # Re-fit GMM
    print(f"\nFitting GMM (K={cfg.preference.K}, iters={args.fit_iters}, lr={args.fit_lr})...")
    agent.preference.update_from_latents(
        latent_tensor,
        n_iters=args.fit_iters,
        lr=args.fit_lr,
    )

    # Report GMM health
    log_prob = agent.preference.log_prob(latent_tensor).mean().item()
    weights = torch.softmax(agent.preference.logits.data, dim=0)
    means = agent.preference.means.data
    min_dist = float("inf")
    for i in range(means.shape[0]):
        for j in range(i + 1, means.shape[0]):
            d = (means[i] - means[j]).norm().item()
            if d < min_dist:
                min_dist = d

    print(f"\nGMM Health:")
    print(f"  log_prob: {log_prob:.2f}")
    print(f"  weights:  {weights.tolist()}")
    print(f"  means norm: {[round(m.norm().item(), 4) for m in means]}")
    print(f"  min_component_dist: {min_dist:.4f}")

    # Test EFE with new preference model
    state = wm.rssm.initial(1, device)
    horizon = cfg.cem.horizon
    for name, accel_val in [("accelerate", 0.5), ("brake", -0.5), ("zero", 0.0)]:
        acts = torch.zeros(1, horizon, 2, device=device)
        acts[:, :, 1] = accel_val
        expanded = type(state)(*[x.expand(1, -1) for x in state])
        traj = wm.rssm.imagine(expanded, acts.permute(1, 0, 2))
        feats = [wm.rssm.get_feat(s) for s in traj]
        means_l = [s.mean for s in traj]
        stds_l = [s.std for s in traj]
        efe = agent.efe_scorer.score(feats, means_l, stds_l, agent.preference, wm.ensemble)
        print(f"  {name:12s}: EFE={efe.item():.4f}")

    # Save updated checkpoint
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(args.output)
    print(f"\nSaved refit checkpoint: {args.output}")


if __name__ == "__main__":
    main()

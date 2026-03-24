"""Re-fit the GMM preference model using only high-speed (driving) frames.

The original GMM included standing-still frames, causing the agent to
prefer braking. This script encodes preference data through the world
model, filters to only moving frames (speed > threshold), and re-fits
the GMM. The world model weights are NOT changed.

Supports task-specific filtering via --task_filter:
  A = lane-keeping only (task_label==0)
  B = lane-change only (task_label==1)
  (omit for all data)

Usage:
    uv run python scripts/refit_preference.py \
        --checkpoint outputs/train_v1/checkpoints/final.pt \
        --data data/expert_data.h5 \
        --output outputs/train_v1/checkpoints/final_refit.pt \
        --min_speed 1.0

    # Task B preference (lane-change episodes only):
    uv run python scripts/refit_preference.py \
        --checkpoint outputs/train_v6_combined/checkpoints/best.pt \
        --data data/expert_data_v6_combined.h5 \
        --output outputs/train_v6_combined/checkpoints/best_taskb_pref.pt \
        --task_filter B --K 7
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
    parser.add_argument("--task_filter", choices=["A", "B"], default=None,
                        help="Filter by task: A=lane-keep, B=lane-change")
    parser.add_argument("--K", type=int, default=None,
                        help="Override GMM component count (default: use config)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # Determine target K for refitting
    target_K = args.K if args.K is not None else cfg.preference.K

    # Probe checkpoint to get its preference K (may differ from config)
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_K = ckpt_peek["preference"]["means"].shape[0]
    del ckpt_peek

    # Build agent with checkpoint's K so load_checkpoint succeeds
    cfg.preference.K = ckpt_K
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    # Reinitialize preference with target K for refitting
    if target_K != ckpt_K:
        print(f"Overriding GMM K: {ckpt_K} (checkpoint) -> {target_K} (target)")
    from active_inference.training.preference import PreferenceModel
    cfg.preference.K = target_K
    agent.preference = PreferenceModel(
        K=target_K,
        latent_dim=cfg.rssm.stoch_dim,
        min_std=cfg.preference.min_std,
    ).to(agent._device)

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
        # Load task_labels for task filtering
        if "task_labels" in f:
            task_labels = f["task_labels"][:]
        else:
            task_labels = None

    print(f"Loaded {len(images)} frames")
    print(f"States shape: {states.shape}")

    # Slice state to match model's expected dim
    if states.shape[1] > state_dim:
        states = states[:, :state_dim]

    # Filter: only successful frames with speed > min_speed
    speeds = states[:, 0]  # speed is first state dimension
    mask = success_flags & (speeds > args.min_speed)

    # Apply task filter
    if args.task_filter is not None:
        if task_labels is None:
            print("WARNING: --task_filter specified but dataset has no task_labels. "
                  "Ignoring filter.")
        else:
            target_label = 0 if args.task_filter == "A" else 1
            task_mask = task_labels == target_label
            mask = mask & task_mask
            label_name = "lane-keep" if args.task_filter == "A" else "lane-change"
            n_task = int(task_mask.sum())
            print(f"Task filter: {args.task_filter} ({label_name}) -> "
                  f"{n_task}/{len(images)} frames match")

    valid_indices = np.where(mask)[0]
    print(f"Frames after all filters: {len(valid_indices)}/{len(images)}")

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

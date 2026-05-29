"""Analyze 5D training and evaluation results.

Checks:
1. Training loss convergence
2. State decoder obstacle_distance prediction quality
3. Eval metrics (obstacle avoidance, lane changes, completion)
"""

import csv
import sys
from pathlib import Path

import torch
import numpy as np


def analyze_training(train_dir: Path):
    """Check training checkpoints and loss."""
    print("=" * 50)
    print("TRAINING ANALYSIS")
    print("=" * 50)

    ckpt_dir = train_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    best = ckpt_dir / "best.pt"
    final = ckpt_dir / "final.pt"

    print(f"Checkpoints found: {len(checkpoints)}")
    if best.exists():
        print(f"  best.pt: {best.stat().st_size / 1e6:.1f} MB")
    if final.exists():
        print(f"  final.pt: {final.stat().st_size / 1e6:.1f} MB")

    # Load best checkpoint and check state decoder
    if best.exists():
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        wm = ckpt["world_model"]
        # Check state decoder output dim
        for key in wm:
            if "state_decoder" in key:
                print(f"  {key}: {wm[key].shape}")
        print()


def analyze_state_decoder_quality(checkpoint_path: Path, data_path: Path):
    """Check if state decoder can predict obstacle_distance."""
    print("=" * 50)
    print("STATE DECODER QUALITY (obstacle_distance prediction)")
    print("=" * 50)

    try:
        import h5py
        from active_inference.config import Config
        from active_inference.agent import DeepAIFAgent

        cfg = Config.from_yaml("configs/experiment/task_b.yaml")
        # Probe checkpoint K
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_K = ckpt["preference"]["means"].shape[0]
        cfg.preference.K = ckpt_K
        del ckpt

        agent = DeepAIFAgent(cfg)
        agent.load_checkpoint(str(checkpoint_path))
        wm = agent.world_model
        device = agent._device

        with h5py.File(data_path, "r") as f:
            images = f["images"][:500]
            states = f["states"][:500]
            actions = f["actions"][:500]
            task_labels = f["task_labels"][:500] if "task_labels" in f else None

        # Encode and decode
        true_obs_dist = states[:, 4] if states.shape[1] >= 5 else None
        if true_obs_dist is None:
            print("  No 5th state dimension in data")
            return

        pred_obs_dists = []
        state = wm.rssm.initial(1, device)
        prev_act = torch.zeros(1, cfg.cem.action_dim, device=device)

        with torch.no_grad():
            for i in range(min(500, len(images))):
                img = torch.tensor(images[i], dtype=torch.float32).unsqueeze(0).to(device)
                st = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                embed = wm.encode_obs(img, st)
                post, _ = wm.rssm.obs_step(state, prev_act, embed)
                feat = wm.rssm.get_feat(post)
                decoded = wm.state_decoder(feat)
                pred_obs_dists.append(decoded[0, 4].item())
                from active_inference.models.rssm import RSSMState
                state = type(post)(*[x.detach() for x in post])
                prev_act = torch.tensor(actions[i], dtype=torch.float32).unsqueeze(0).to(device)

        pred = np.array(pred_obs_dists)
        true = true_obs_dist[:len(pred)]

        # Compute metrics
        mae = np.abs(pred - true).mean()
        corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else 0

        # Task B specific (near obstacles)
        if task_labels is not None:
            tb_mask = task_labels[:len(pred)] == 1
            if tb_mask.sum() > 0:
                tb_mae = np.abs(pred[tb_mask] - true[tb_mask]).mean()
                tb_corr = np.corrcoef(pred[tb_mask], true[tb_mask])[0, 1] if tb_mask.sum() > 1 else 0
            else:
                tb_mae = tb_corr = float("nan")
        else:
            tb_mae = tb_corr = float("nan")

        # Near-obstacle accuracy (true < 0.5)
        near_mask = true < 0.5
        if near_mask.sum() > 0:
            near_mae = np.abs(pred[near_mask] - true[near_mask]).mean()
            near_corr = np.corrcoef(pred[near_mask], true[near_mask])[0, 1] if near_mask.sum() > 1 else 0
        else:
            near_mae = near_corr = float("nan")

        print(f"  Overall: MAE={mae:.4f}, corr={corr:.4f} (n={len(pred)})")
        print(f"  Task B:  MAE={tb_mae:.4f}, corr={tb_corr:.4f} (n={int(tb_mask.sum()) if task_labels is not None else 0})")
        print(f"  Near obs: MAE={near_mae:.4f}, corr={near_corr:.4f} (n={int(near_mask.sum())})")
        print(f"  Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"  True range: [{true.min():.3f}, {true.max():.3f}]")
        print()
    except Exception as e:
        print(f"  Error: {e}")
        print()


def analyze_eval_results(eval_dir: Path, task_name: str):
    """Parse eval CSV results."""
    print("=" * 50)
    print(f"EVALUATION: {task_name}")
    print("=" * 50)

    csv_path = eval_dir / "eval_results.csv"
    if not csv_path.exists():
        print(f"  No results at {csv_path}")
        return

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  Empty CSV")
        return

    completions = [float(r.get("completion", 0)) for r in rows]
    mlds = [float(r.get("mld", 0)) for r in rows]

    print(f"  Episodes: {len(rows)}")
    print(f"  Avg completion: {np.mean(completions)*100:.1f}%")
    print(f"  Avg MLD: {np.mean(mlds):.3f}")

    if "obstacles_avoided" in rows[0]:
        total_avoided = sum(int(r.get("obstacles_avoided", 0)) for r in rows)
        total_obstacles = sum(int(r.get("total_obstacles", 0)) for r in rows)
        avoidance_rate = total_avoided / max(total_obstacles, 1) * 100
        total_lc = sum(int(r.get("lane_changes", 0)) for r in rows)
        print(f"  Obstacles avoided: {total_avoided}/{total_obstacles} ({avoidance_rate:.0f}%)")
        print(f"  Total lane changes: {total_lc}")

    print()
    print("  Per-episode:")
    for r in rows:
        route = r.get("route_index", "?")
        ep = r.get("episode", "?")
        comp = float(r.get("completion", 0)) * 100
        mld = float(r.get("mld", 0))
        term = r.get("termination", "?")
        avoided = r.get("obstacles_avoided", "")
        total = r.get("total_obstacles", "")
        lc = r.get("lane_changes", "")
        obs_str = f" obs={avoided}/{total}" if avoided else ""
        lc_str = f" lc={lc}" if lc else ""
        print(f"    R{route}E{ep}: {comp:.0f}% MLD={mld:.3f}{obs_str}{lc_str} [{term}]")
    print()


def main():
    train_dir = Path("outputs/train_v7_5d")
    data_path = Path("data/expert_data_v7_5d.h5")
    eval_b_dir = Path("outputs/eval_task_b_v4_5d")
    eval_a_dir = Path("outputs/eval_task_a_v7")

    analyze_training(train_dir)

    best_ckpt = train_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists() and data_path.exists():
        analyze_state_decoder_quality(best_ckpt, data_path)

    if eval_b_dir.exists():
        analyze_eval_results(eval_b_dir, "Task B (Obstacle Avoidance)")

    if eval_a_dir.exists():
        analyze_eval_results(eval_a_dir, "Task A Regression (Lane Keeping)")


if __name__ == "__main__":
    main()

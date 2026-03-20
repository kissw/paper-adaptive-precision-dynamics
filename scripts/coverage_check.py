import argparse
import sys

import numpy as np
import h5py


def compute_coverage(actions: np.ndarray, bins: int = 20) -> dict:
    H, xedges, yedges = np.histogram2d(
        actions[:, 0],
        actions[:, 1],
        bins=bins,
        range=[[-1, 1], [-1, 1]],
    )
    occupied = (H > 10).sum()
    total = H.size
    coverage = occupied / total

    H_norm = H / H.sum()
    H_safe = H_norm + 1e-10
    entropy = -np.sum(H_safe * np.log(H_safe))
    max_entropy = np.log(total)

    return {
        "bins": bins,
        "occupied_bins": int(occupied),
        "total_bins": total,
        "coverage": coverage,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy,
    }


def main():
    parser = argparse.ArgumentParser(description="Check action-space coverage of collected data")
    parser.add_argument("data", help="HDF5 data file path")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--min_coverage", type=float, default=0.30)
    parser.add_argument("--min_entropy", type=float, default=3.5)
    args = parser.parse_args()

    with h5py.File(args.data, "r") as f:
        actions = f["actions"][:]
        n_frames = len(actions)

        has_expert = "expert_actions" in f
        has_meta = "success_flags" in f

        print(f"Dataset: {args.data}")
        print(f"Frames: {n_frames}")
        if "total_episodes" in f.attrs:
            print(f"Episodes: {f.attrs['total_episodes']}")
        if "tier_distribution" in f.attrs:
            print(f"Tiers: {f.attrs['tier_distribution']}")

        if has_meta:
            success = f["success_flags"][:]
            task_labels = f["task_labels"][:]
            print(f"Success rate: {success.sum()}/{n_frames} ({100 * success.mean():.1f}%)")
            task_a = (task_labels == 0).sum()
            task_b = (task_labels == 1).sum()
            print(f"Task A (lane keep): {task_a} frames")
            print(f"Task B (lane change): {task_b} frames")

    print(f"\n--- Action Coverage (bins={args.bins}) ---")

    result = compute_coverage(actions, bins=args.bins)
    print(f"Occupied bins (>10 samples): {result['occupied_bins']}/{result['total_bins']}")
    print(f"Coverage: {result['coverage']:.3f} (target >= {args.min_coverage})")
    print(f"Entropy: {result['entropy']:.3f} (target >= {args.min_entropy})")
    print(f"Normalized entropy: {result['normalized_entropy']:.3f}")

    print(f"\n--- Per-dimension stats ---")
    print(
        f"Steer:  mean={actions[:, 0].mean():.4f}  std={actions[:, 0].std():.4f}  "
        f"min={actions[:, 0].min():.3f}  max={actions[:, 0].max():.3f}"
    )
    print(
        f"Accel:  mean={actions[:, 1].mean():.4f}  std={actions[:, 1].std():.4f}  "
        f"min={actions[:, 1].min():.3f}  max={actions[:, 1].max():.3f}"
    )

    if has_expert:
        with h5py.File(args.data, "r") as f:
            expert = f["expert_actions"][:]
        expert_result = compute_coverage(expert, bins=args.bins)
        print(f"\n--- Expert (unperturbed) coverage ---")
        print(f"Coverage: {expert_result['coverage']:.3f}")
        print(f"Entropy: {expert_result['entropy']:.3f}")
        print(f"Noise improvement: {result['entropy'] - expert_result['entropy']:.3f} bits")

    ok = result["coverage"] >= args.min_coverage and result["entropy"] >= args.min_entropy
    print(
        f"\n{'PASS' if ok else 'FAIL'}: coverage={'OK' if result['coverage'] >= args.min_coverage else 'LOW'}, "
        f"entropy={'OK' if result['entropy'] >= args.min_entropy else 'LOW'}"
    )

    if not ok:
        print("Recommendation: collect more high-noise episodes to improve coverage.")
        sys.exit(1)


if __name__ == "__main__":
    main()

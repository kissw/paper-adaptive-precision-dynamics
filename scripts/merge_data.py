"""Merge multiple HDF5 driving datasets into one for training.

Usage:
    uv run python scripts/merge_data.py \
        data/expert_data_v4.h5 data/expert_data_town04.h5 \
        --output data/expert_data_mixed.h5
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge HDF5 driving datasets")
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    args = parser.parse_args()

    all_data = {}
    total_frames = 0
    total_episodes = 0
    towns = []

    for path in args.inputs:
        print(f"Reading {path}...")
        with h5py.File(path, "r") as f:
            n = f["images"].shape[0]
            state_dim = f["states"].shape[1]
            town = f.attrs.get("town", "unknown")
            eps = f.attrs.get("total_episodes", 0)
            towns.append(f"{town}({n})")
            print(f"  {n} frames, {eps} episodes, state_dim={state_dim}, town={town}")

            for key in f.keys():
                arr = f[key][:]
                if key == "episode_ids":
                    arr = arr + total_episodes
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(arr)

            total_frames += n
            total_episodes += eps

    # Ensure all state arrays have the same dimension
    # Pad with 1.0 for obstacle_distance (5th dim): 1.0 = no obstacle nearby
    state_arrays = all_data["states"]
    max_dim = max(s.shape[1] for s in state_arrays)
    padded = []
    for s in state_arrays:
        if s.shape[1] < max_dim:
            orig_dim = s.shape[1]
            pad = np.ones((s.shape[0], max_dim - orig_dim), dtype=s.dtype)
            s = np.concatenate([s, pad], axis=1)
            print(f"  Padded states from {orig_dim} to {max_dim}D (1.0 for obstacle_distance)")
        padded.append(s)
    all_data["states"] = padded

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        for key, arrays in all_data.items():
            merged = np.concatenate(arrays, axis=0)
            f.create_dataset(key, data=merged, dtype=merged.dtype)
            print(f"  {key}: {merged.shape}")

        f.attrs["town"] = "+".join(towns)
        f.attrs["total_frames"] = total_frames
        f.attrs["total_episodes"] = total_episodes

    print(f"\nMerged {total_frames} frames ({total_episodes} episodes) -> {args.output}")
    print(f"Sources: {', '.join(towns)}")


if __name__ == "__main__":
    main()

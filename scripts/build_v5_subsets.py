"""Build four HDF5 subset files from v5 obstacle and clean data.

Inputs
------
  --obstacle_data   data/expert_obstacle_v5.h5
  --clean_data      data/expert_clean_v5.h5

Outputs
-------
  1. data/world_model_train_v5.h5
       All frames from obstacle + clean, merged.
       episode_ids from clean are offset by max(obstacle_episode_ids)+1 to
       prevent collisions.  collection_type = "world_model_train_v5"

  2. data/clean_preference_v5.h5
       70 % of clean episodes (episode-level split, seed-fixed).
       collection_type = "clean_preference_v5"

  3. data/obstacle_visible_train_v5.h5
       obstacle_visible==True frames from the 70 % obstacle episodes.
       collection_type = "obstacle_visible_train_v5"

  4. data/obstacle_visible_eval_v5.h5
       Two HDF5 groups inside:
         /clean           : frames from 30 % clean episodes
         /obstacle_visible: visible frames from 30 % obstacle episodes
       collection_type = "pp_gap_eval_v5"

Split policy
-----------
  * Episode-level 70/30 split (not frame-level).
  * Seed fixed (CLI --seed, default 42).
  * train_ratio adjustable via --train_ratio.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_h5(path: str) -> dict:
    """Load all datasets from an HDF5 file into a dict of numpy arrays."""
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
        data["__attrs__"] = dict(f.attrs)
    return data


def episode_split(episode_ids: np.ndarray, train_ratio: float, seed: int) -> tuple:
    """Split unique episode IDs into train/eval sets.

    Returns (train_ids_set, eval_ids_set).
    """
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_eps)
    n_train = max(1, int(len(shuffled) * train_ratio))
    train_ids = set(shuffled[:n_train].tolist())
    eval_ids = set(shuffled[n_train:].tolist())
    return train_ids, eval_ids


def select_by_episode(data: dict, episode_ids_set: set) -> dict:
    """Select all frames whose episode_id is in episode_ids_set."""
    mask = np.isin(data["episode_ids"], list(episode_ids_set))
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v
            for k, v in data.items() if k != "__attrs__"}


def select_by_mask(data: dict, mask: np.ndarray) -> dict:
    """Select frames by a boolean mask."""
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v
            for k, v in data.items() if k != "__attrs__"}


def write_h5(path: str, data: dict, attrs: dict):
    """Write dict of arrays to HDF5, then apply attrs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr)
        for k, v in attrs.items():
            f.attrs[k] = v


def write_h5_groups(path: str, groups: dict, attrs: dict):
    """Write named groups to HDF5; each group is a dict of arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for grp_name, grp_data in groups.items():
            grp = f.create_group(grp_name)
            for key, arr in grp_data.items():
                grp.create_dataset(key, data=arr)
        for k, v in attrs.items():
            f.attrs[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(a: dict, b: dict, b_episode_id_offset: int) -> dict:
    """Merge two frame-dicts; offset b's episode_ids to avoid collisions."""
    b_shifted = dict(b)
    b_shifted["episode_ids"] = b["episode_ids"] + b_episode_id_offset
    merged = {}
    all_keys = set(a.keys()) | set(b_shifted.keys())
    for key in all_keys:
        if key in a and key in b_shifted:
            merged[key] = np.concatenate([a[key], b_shifted[key]], axis=0)
        elif key in a:
            merged[key] = a[key]
        else:
            merged[key] = b_shifted[key]
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build four v5 HDF5 subsets from obstacle + clean data",
    )
    parser.add_argument("--obstacle_data", default="data/expert_obstacle_v5.h5")
    parser.add_argument("--clean_data", default="data/expert_clean_v5.h5")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading obstacle data from {args.obstacle_data} …")
    obs = load_h5(args.obstacle_data)
    print(f"  {len(obs['episode_ids'])} frames, "
          f"{len(np.unique(obs['episode_ids']))} episodes")

    print(f"Loading clean data from {args.clean_data} …")
    cln = load_h5(args.clean_data)
    print(f"  {len(cln['episode_ids'])} frames, "
          f"{len(np.unique(cln['episode_ids']))} episodes")

    # ── Episode-level splits ──────────────────────────────────────────────
    obs_train_eps, obs_eval_eps = episode_split(
        obs["episode_ids"], args.train_ratio, args.seed,
    )
    cln_train_eps, cln_eval_eps = episode_split(
        cln["episode_ids"], args.train_ratio, args.seed,
    )

    obs_train = select_by_episode(obs, obs_train_eps)
    obs_eval  = select_by_episode(obs, obs_eval_eps)
    cln_train = select_by_episode(cln, cln_train_eps)
    cln_eval  = select_by_episode(cln, cln_eval_eps)

    # ── 1. world_model_train_v5.h5 ────────────────────────────────────────
    ep_offset = int(obs["episode_ids"].max()) + 1
    wm_train = merge_datasets(obs, cln, b_episode_id_offset=ep_offset)
    out1 = str(Path(args.output_dir) / "world_model_train_v5.h5")
    write_h5(out1, wm_train, {
        "collection_type": "world_model_train_v5",
        "total_frames": len(wm_train["episode_ids"]),
        "episode_id_offset_clean": ep_offset,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    })
    print(f"[1] world_model_train_v5.h5 → {len(wm_train['episode_ids'])} frames")

    # ── 2. clean_preference_v5.h5 ────────────────────────────────────────
    out2 = str(Path(args.output_dir) / "clean_preference_v5.h5")
    write_h5(out2, cln_train, {
        "collection_type": "clean_preference_v5",
        "total_frames": len(cln_train["episode_ids"]),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    })
    print(f"[2] clean_preference_v5.h5 → {len(cln_train['episode_ids'])} frames")

    # ── 3. obstacle_visible_train_v5.h5 ──────────────────────────────────
    vis_mask = obs_train["obstacle_visible"].astype(bool)
    obs_vis_train = select_by_mask(obs_train, vis_mask)
    out3 = str(Path(args.output_dir) / "obstacle_visible_train_v5.h5")
    write_h5(out3, obs_vis_train, {
        "collection_type": "obstacle_visible_train_v5",
        "total_frames": len(obs_vis_train["episode_ids"]),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    })
    print(
        f"[3] obstacle_visible_train_v5.h5 → "
        f"{len(obs_vis_train['episode_ids'])} visible frames"
    )

    # ── 4. obstacle_visible_eval_v5.h5 ───────────────────────────────────
    vis_eval_mask = obs_eval["obstacle_visible"].astype(bool)
    obs_vis_eval = select_by_mask(obs_eval, vis_eval_mask)
    out4 = str(Path(args.output_dir) / "obstacle_visible_eval_v5.h5")
    write_h5_groups(
        out4,
        {
            "clean": cln_eval,
            "obstacle_visible": obs_vis_eval,
        },
        {
            "collection_type": "pp_gap_eval_v5",
            "clean_frames": len(cln_eval["episode_ids"]),
            "obstacle_visible_frames": len(obs_vis_eval["episode_ids"]),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
    )
    print(
        f"[4] obstacle_visible_eval_v5.h5 → "
        f"clean:{len(cln_eval['episode_ids'])} "
        f"obs_vis:{len(obs_vis_eval['episode_ids'])}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (used by tests and pp_gap_v5.py)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "load_h5",
    "episode_split",
    "select_by_episode",
    "select_by_mask",
    "merge_datasets",
    "write_h5",
    "write_h5_groups",
]

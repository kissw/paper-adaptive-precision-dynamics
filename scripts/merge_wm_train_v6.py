"""Merge continuous expert data (v5) and counterfactual data (v6) for world-model training.

Produces a single HDF5 with:
- All expert frames
- cf_ratio fraction of counterfactual frames (random sample, reproducible via --seed)
- episode_ids offset so expert and CF episodes do not collide
- attrs recording provenance and merge parameters

Usage:
    uv run python scripts/merge_wm_train_v6.py \
        --expert_data  data/world_model_train_v5.h5 \
        --cf_data      data/counterfactual_v6_main.h5 \
        --output       data/wm_train_v6_merged.h5 \
        --cf_ratio     0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REQUIRED_KEYS = ["images", "states", "actions", "episode_ids"]
WORLD_MODEL_KEYS = [
    "images", "states", "actions", "episode_ids",
    "obstacle_visible", "obstacle_distance", "obstacle_in_front",
    "obstacle_lateral", "obstacle_bbox", "obstacle_bbox_area",
    "ego_x", "ego_y", "ego_yaw", "obstacle_x", "obstacle_y",
]


def check_schema_compatibility(expert_path: str | Path, cf_path: str | Path) -> None:
    """Raise ValueError if either file is missing any of REQUIRED_KEYS."""
    for label, path in [("expert", expert_path), ("cf", cf_path)]:
        with h5py.File(path, "r") as f:
            missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise ValueError(
                f"{label} file {path} is missing required keys: {missing}"
            )


def load_h5_keys(path: str | Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Load the intersection of `keys` and keys present in the file."""
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                result[k] = f[k][:]
    return result


def sample_cf_frames(
    cf_data: dict[str, np.ndarray],
    cf_ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Randomly select cf_ratio fraction of CF frames (whole-index sample)."""
    if cf_ratio <= 0.0:
        n_select = 0
    elif cf_ratio >= 1.0:
        return {k: v.copy() for k, v in cf_data.items()}
    else:
        n_total = len(cf_data["images"])
        n_select = max(1, int(round(n_total * cf_ratio)))

    if n_select == 0:
        return {k: v[:0] for k, v in cf_data.items()}

    n_total = len(cf_data["images"])
    idx = rng.choice(n_total, size=n_select, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in cf_data.items()}


def merge_datasets(
    expert_path: str | Path,
    cf_path: str | Path,
    output_path: str | Path,
    *,
    cf_ratio: float = 1.0,
    seed: int = 0,
    extra_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Merge expert + CF data and write to output_path.

    Returns a dict of summary stats written into the output attrs.
    """
    check_schema_compatibility(expert_path, cf_path)

    keys = list(WORLD_MODEL_KEYS)
    if extra_keys:
        keys = keys + [k for k in extra_keys if k not in keys]

    expert_data = load_h5_keys(expert_path, keys)
    cf_data_full = load_h5_keys(cf_path, keys)

    rng = np.random.default_rng(seed)
    cf_data = sample_cf_frames(cf_data_full, cf_ratio, rng)

    n_expert = len(expert_data["images"])
    n_cf = len(cf_data.get("images", []))

    # Offset CF episode_ids to avoid collision with expert episode_ids
    expert_max_ep = int(expert_data["episode_ids"].max()) if n_expert > 0 else -1
    ep_offset = expert_max_ep + 1
    if n_cf > 0:
        cf_data["episode_ids"] = cf_data["episode_ids"] + ep_offset

    # Build merged arrays over keys present in at least one dataset
    all_keys = set(expert_data.keys()) | set(cf_data.keys())
    merged: dict[str, np.ndarray] = {}
    for k in all_keys:
        parts = []
        if k in expert_data and len(expert_data[k]) > 0:
            parts.append(expert_data[k])
        if k in cf_data and len(cf_data[k]) > 0:
            parts.append(cf_data[k])
        if parts:
            merged[k] = np.concatenate(parts, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for k, arr in merged.items():
            f.create_dataset(k, data=arr)

        attrs = {
            "expert_path": str(expert_path),
            "cf_path": str(cf_path),
            "cf_ratio": float(cf_ratio),
            "seed": int(seed),
            "n_expert_frames": int(n_expert),
            "n_cf_frames_total": int(len(cf_data_full.get("images", []))),
            "n_cf_frames_sampled": int(n_cf),
            "total_frames": int(len(merged.get("images", []))),
            "episode_id_offset": int(ep_offset),
        }
        for k, v in attrs.items():
            f.attrs[k] = v

    return attrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge expert + CF data for WM training")
    parser.add_argument("--expert_data", required=True, help="Path to v5 expert HDF5")
    parser.add_argument("--cf_data", required=True, help="Path to v6 counterfactual HDF5")
    parser.add_argument("--output", required=True, help="Output merged HDF5 path")
    parser.add_argument(
        "--cf_ratio", type=float, default=1.0,
        help="Fraction of CF frames to include (0.0–1.0, default=1.0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for CF sampling")
    args = parser.parse_args()

    print(f"expert_data : {args.expert_data}")
    print(f"cf_data     : {args.cf_data}")
    print(f"output      : {args.output}")
    print(f"cf_ratio    : {args.cf_ratio}")
    print(f"seed        : {args.seed}")

    summary = merge_datasets(
        args.expert_data,
        args.cf_data,
        args.output,
        cf_ratio=args.cf_ratio,
        seed=args.seed,
    )

    print("\nMerge complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

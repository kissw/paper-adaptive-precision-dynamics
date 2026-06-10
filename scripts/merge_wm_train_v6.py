"""Merge continuous expert data (v5) and counterfactual data (v6) for world-model training.

Produces a single HDF5 with:
- All expert frames
- cf_ratio fraction of counterfactual *episodes* (episode-level selection, keeps
  each context+branch sequence intact so dataset.py valid_starts work correctly)
- episode_ids offset so expert and CF episodes do not collide
- CF identification flags preserved; expert side zero-padded for missing cf_* keys
- attrs recording provenance and merge parameters

Usage:
    uv run python scripts/merge_wm_train_v6.py \\
        --expert_data  data/world_model_train_v5.h5 \\
        --cf_data      data/counterfactual_v6_main.h5 \\
        --output       data/wm_train_v6_merged.h5 \\
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
# CF identification and diagnostic flags written into the merged output.
# Expert side is zero-padded; CF side is loaded from file (or synthesised).
CF_FLAG_KEYS = [
    "is_counterfactual",
    "cf_is_branch", "cf_is_context",
    "cf_anchor_id",                    # needed for post-merge anchor leakage check
    "cf_branch_step", "cf_steer_bias",
    "cf_branch_collision_diag", "cf_branch_lane_invasion_diag",
]


def check_schema_compatibility(expert_path: str | Path, cf_path: str | Path) -> None:
    """Raise ValueError if either file is missing REQUIRED_KEYS or shapes mismatch."""
    for label, path in [("expert", expert_path), ("cf", cf_path)]:
        with h5py.File(path, "r") as f:
            missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise ValueError(
                f"{label} file {path} is missing required keys: {missing}"
            )

    with h5py.File(expert_path, "r") as ef, h5py.File(cf_path, "r") as cf:
        for key in ("states", "actions"):
            if key in ef and key in cf:
                ed, cd = ef[key].shape[-1], cf[key].shape[-1]
                if ed != cd:
                    raise ValueError(
                        f"{key} last-dim mismatch: expert={ed}, cf={cd}"
                    )
        if "images" in ef and "images" in cf:
            es, cs = ef["images"].shape[1:], cf["images"].shape[1:]
            if es != cs:
                raise ValueError(
                    f"images shape mismatch: expert={es}, cf={cs}"
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
    """Select cf_ratio fraction of CF episodes (episode-level, sequence-preserving).

    Episode-level selection keeps each context+branch sequence intact so that
    dataset.py valid_starts (which require seq_len consecutive same-episode frames)
    remain valid. Frame-level random selection would break this invariant.
    """
    if cf_ratio >= 1.0:
        return {k: v.copy() for k, v in cf_data.items()}
    if cf_ratio <= 0.0:
        return {k: v[:0] for k, v in cf_data.items()}

    eids = cf_data["episode_ids"]
    unique_eps = np.unique(eids)
    n_select = max(1, int(round(len(unique_eps) * cf_ratio)))
    chosen = rng.choice(unique_eps, size=n_select, replace=False)
    chosen_set = set(chosen.tolist())

    # Keep all frames belonging to chosen episodes in original order.
    mask = np.isin(eids, list(chosen_set))
    return {k: v[mask] for k, v in cf_data.items()}


def _pad_missing_keys(
    primary: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    n: int,
) -> None:
    """Add zero-filled arrays to `primary` for keys present only in `reference`."""
    for k, ref_arr in reference.items():
        if k not in primary and n > 0:
            shape = (n,) + ref_arr.shape[1:]
            primary[k] = np.zeros(shape, dtype=ref_arr.dtype)


def merge_datasets(
    expert_path: str | Path,
    cf_path: str | Path,
    output_path: str | Path,
    *,
    cf_ratio: float = 1.0,
    seed: int = 0,
    extra_keys: list[str] | None = None,
    drop_collision_branches: bool = False,
) -> dict[str, Any]:
    """Merge expert + CF data and write to output_path.

    Returns a dict of summary stats written into the output attrs.
    """
    check_schema_compatibility(expert_path, cf_path)

    keys = list(WORLD_MODEL_KEYS) + [k for k in CF_FLAG_KEYS if k not in WORLD_MODEL_KEYS]
    if extra_keys:
        keys = keys + [k for k in extra_keys if k not in keys]

    expert_data = load_h5_keys(expert_path, keys)
    cf_data_full = load_h5_keys(cf_path, keys)
    n_cf_frames_total = len(cf_data_full.get("images", []))

    # Optional: drop CF episodes that contain any collision in the diagnostic window.
    n_cf_before_filter = n_cf_frames_total
    if drop_collision_branches and "cf_branch_collision_diag" in cf_data_full:
        collision = cf_data_full["cf_branch_collision_diag"].astype(bool)
        eids_cf = cf_data_full["episode_ids"]
        collision_eps = set(eids_cf[collision].tolist())
        keep_mask = ~np.isin(eids_cf, list(collision_eps))
        cf_data_full = {k: v[keep_mask] for k, v in cf_data_full.items()}
        n_cf_frames_total = len(cf_data_full.get("images", []))

    rng = np.random.default_rng(seed)
    cf_data = sample_cf_frames(cf_data_full, cf_ratio, rng)

    n_expert = len(expert_data.get("images", []))
    n_cf = len(cf_data.get("images", []))

    # ── is_counterfactual flag ──────────────────────────────────────────────
    # Expert frames are never counterfactual; CF frames always are.
    expert_data["is_counterfactual"] = np.zeros(n_expert, dtype=bool)
    if n_cf > 0:
        if "is_counterfactual" not in cf_data:
            cf_data["is_counterfactual"] = np.ones(n_cf, dtype=bool)
        else:
            cf_data["is_counterfactual"] = cf_data["is_counterfactual"].astype(bool)

    # ── pad missing keys with zeros so concatenation has equal lengths ──────
    _pad_missing_keys(expert_data, cf_data, n_expert)
    _pad_missing_keys(cf_data, expert_data, n_cf)

    # ── episode_id offset: CF episodes must not collide with expert IDs ─────
    expert_max_ep = int(expert_data["episode_ids"].max()) if n_expert > 0 else -1
    ep_offset = expert_max_ep + 1
    if n_cf > 0:
        cf_data["episode_ids"] = cf_data["episode_ids"] + ep_offset

    # ── concatenate ─────────────────────────────────────────────────────────
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

        attrs: dict[str, Any] = {
            "expert_path": str(expert_path),
            "cf_path": str(cf_path),
            "cf_ratio": float(cf_ratio),
            "seed": int(seed),
            "drop_collision_branches": bool(drop_collision_branches),
            "n_expert_frames": int(n_expert),
            "n_cf_frames_total": int(n_cf_before_filter),
            "n_cf_frames_after_collision_filter": int(n_cf_frames_total),
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
        help="Fraction of CF episodes to include (0.0–1.0, default=1.0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for CF sampling")
    parser.add_argument(
        "--drop_collision_branches", action="store_true", default=False,
        help=(
            "Exclude CF episodes where cf_branch_collision_diag==True. "
            "Default False (keeps strong-maneuver branches for training diversity)."
        ),
    )
    args = parser.parse_args()

    print(f"expert_data              : {args.expert_data}")
    print(f"cf_data                  : {args.cf_data}")
    print(f"output                   : {args.output}")
    print(f"cf_ratio                 : {args.cf_ratio}")
    print(f"seed                     : {args.seed}")
    print(f"drop_collision_branches  : {args.drop_collision_branches}")

    summary = merge_datasets(
        args.expert_data,
        args.cf_data,
        args.output,
        cf_ratio=args.cf_ratio,
        seed=args.seed,
        drop_collision_branches=args.drop_collision_branches,
    )

    print("\nMerge complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

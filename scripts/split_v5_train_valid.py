#!/usr/bin/env python3
"""Episode-level train/valid splitter for merged PAActInf v5 HDF5 data.

Given a merged v5 dataset such as data/expert_data_v5.h5, create leakage-safe
episode-level train/valid files for the world model and preference fitting.

Outputs
-------
output_dir/
    world_model_train_v5.h5
    world_model_valid_v5.h5
    clean_preference_train_v5.h5
    clean_preference_valid_v5.h5
    obstacle_visible_preference_train_v5.h5
    obstacle_visible_preference_valid_v5.h5
    pp_gap_eval_v5.h5
    split_metadata_v5.h5

Example
-------
uv run python scripts/split_v5_train_valid.py \
    --input data/expert_data_v5.h5 \
    --output_dir data/v5_split \
    --train_ratio 0.7 \
    --seed 42 \
    --seq_len 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_h5_flat(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load all root-level datasets from a flat HDF5 file."""
    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = obj[:]
            else:
                raise ValueError(
                    f"Input must be a flat HDF5 file. Found group at key={key!r}."
                )
        attrs = dict(f.attrs)
    return data, attrs


def _validate_input(data: dict[str, np.ndarray]) -> int:
    required = ["images", "states", "actions", "episode_ids"]
    for key in required:
        if key not in data:
            raise KeyError(f"Required key missing from input HDF5: {key}")

    n = int(data["images"].shape[0])
    for key, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.shape and arr.shape[0] != n:
            raise ValueError(
                f"Root dataset length mismatch: key={key}, "
                f"shape={arr.shape}, expected first dim={n}"
            )

    if "task_labels" not in data:
        print("WARNING: task_labels missing. Creating all-zero task_labels.")
        data["task_labels"] = np.zeros(n, dtype=np.int8)

    if "obstacle_visible" not in data:
        print("WARNING: obstacle_visible missing. Creating all-False obstacle_visible.")
        data["obstacle_visible"] = np.zeros(n, dtype=bool)

    if "source_ids" not in data:
        print("WARNING: source_ids missing. Using task_labels as source_ids fallback.")
        data["source_ids"] = data["task_labels"].astype(np.int32)

    return n


def _episode_majority_value(episode_ids: np.ndarray, values: np.ndarray) -> dict[int, int]:
    """Return majority value per episode."""
    out: dict[int, int] = {}
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        vals, counts = np.unique(values[mask], return_counts=True)
        out[int(ep)] = int(vals[np.argmax(counts)])
    return out


def _stratified_episode_split(
    episode_ids: np.ndarray,
    source_ids: np.ndarray,
    task_labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split episodes by (majority_source_id, majority_task_label)."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train_ratio must satisfy 0 < train_ratio < 1.")

    ep_source = _episode_majority_value(episode_ids, source_ids)
    ep_task = _episode_majority_value(episode_ids, task_labels)

    strata: dict[tuple[int, int], list[int]] = {}
    for ep in np.unique(episode_ids):
        key = (ep_source[int(ep)], ep_task[int(ep)])
        strata.setdefault(key, []).append(int(ep))

    rng = np.random.default_rng(seed)
    train_eps: list[int] = []
    valid_eps: list[int] = []

    print("Episode strata:")
    for key in sorted(strata.keys()):
        eps = np.array(strata[key], dtype=np.int64)
        eps = rng.permutation(eps)
        n = len(eps)

        if n == 1:
            n_train = 1
        else:
            n_train = int(round(n * train_ratio))
            n_train = max(1, min(n - 1, n_train))

        tr = eps[:n_train]
        va = eps[n_train:]

        train_eps.extend(tr.tolist())
        valid_eps.extend(va.tolist())

        print(
            f"  source={key[0]} task={key[1]}: "
            f"episodes={n}, train={len(tr)}, valid={len(va)}"
        )

    train_eps_arr = np.array(sorted(train_eps), dtype=np.int64)
    valid_eps_arr = np.array(sorted(valid_eps), dtype=np.int64)

    if len(set(train_eps_arr.tolist()) & set(valid_eps_arr.tolist())) != 0:
        raise RuntimeError("Episode split overlap detected.")

    return train_eps_arr, valid_eps_arr


def _select_by_mask(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """Select all frame-level datasets by mask."""
    n = len(mask)
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.shape and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def _write_h5(
    path: Path,
    data: dict[str, np.ndarray],
    attrs: dict[str, Any],
    compression: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            if not isinstance(arr, np.ndarray):
                continue
            kwargs = {}
            if compression and arr.ndim >= 1 and arr.size > 0:
                kwargs["compression"] = compression
            f.create_dataset(key, data=arr, **kwargs)
        for key, value in attrs.items():
            try:
                f.attrs[key] = value
            except TypeError:
                f.attrs[key] = str(value)


def _write_grouped_h5(
    path: Path,
    groups: dict[str, dict[str, np.ndarray]],
    attrs: dict[str, Any],
    compression: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for group_name, group_data in groups.items():
            g = f.create_group(group_name)
            for key, arr in group_data.items():
                if not isinstance(arr, np.ndarray):
                    continue
                kwargs = {}
                if compression and arr.ndim >= 1 and arr.size > 0:
                    kwargs["compression"] = compression
                g.create_dataset(key, data=arr, **kwargs)
        for key, value in attrs.items():
            try:
                f.attrs[key] = value
            except TypeError:
                f.attrs[key] = str(value)


def _add_common_attrs(
    base_attrs: dict[str, Any],
    split_name: str,
    total_frames: int,
    train_ratio: float,
    seed: int,
    seq_len: int,
) -> dict[str, Any]:
    attrs = dict(base_attrs)
    attrs["split_name"] = split_name
    attrs["total_frames"] = int(total_frames)
    attrs["train_ratio"] = float(train_ratio)
    attrs["split_seed"] = int(seed)
    attrs["seq_len_for_split_check"] = int(seq_len)
    attrs["collection_type"] = split_name
    return attrs


def _count_valid_sequences(episode_ids: np.ndarray, seq_len: int) -> int:
    """Count valid contiguous starts inside same episode."""
    if len(episode_ids) < seq_len:
        return 0
    count = 0
    for i in range(len(episode_ids) - seq_len + 1):
        if episode_ids[i] == episode_ids[i + seq_len - 1]:
            count += 1
    return count


def _print_file_summary(name: str, data: dict[str, np.ndarray], seq_len: int) -> None:
    n = len(data["episode_ids"]) if "episode_ids" in data else 0
    visible = int(data["obstacle_visible"].sum()) if "obstacle_visible" in data else 0
    clean = int((data["task_labels"] == 0).sum()) if "task_labels" in data else -1
    obstacle = int((data["task_labels"] == 1).sum()) if "task_labels" in data else -1
    n_eps = len(np.unique(data["episode_ids"])) if "episode_ids" in data else 0
    n_seq = _count_valid_sequences(data["episode_ids"], seq_len) if "episode_ids" in data else 0

    print(
        f"{name:50s} frames={n:8d} eps={n_eps:4d} "
        f"seq={n_seq:8d} clean={clean:8d} obstacle={obstacle:8d} visible={visible:6d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Episode-level train/valid split for merged PAActInf v5 HDF5 data.",
    )
    parser.add_argument("--input", required=True, help="Merged v5 HDF5 file, e.g. data/expert_data_v5.h5")
    parser.add_argument("--output_dir", required=True, help="Directory for split HDF5 files.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument(
        "--compression",
        default=None,
        choices=[None, "gzip", "lzf"],
        help="Optional HDF5 compression. Default: no compression.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Loading merged v5 dataset: {input_path}")
    data, input_attrs = _read_h5_flat(str(input_path))
    n = _validate_input(data)

    episode_ids = data["episode_ids"].astype(np.int64)
    task_labels = data["task_labels"].astype(np.int32)
    source_ids = data["source_ids"].astype(np.int32)
    obstacle_visible = data["obstacle_visible"].astype(bool)

    train_eps, valid_eps = _stratified_episode_split(
        episode_ids=episode_ids,
        source_ids=source_ids,
        task_labels=task_labels,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    train_mask = np.isin(episode_ids, train_eps)
    valid_mask = np.isin(episode_ids, valid_eps)

    if np.any(train_mask & valid_mask):
        raise RuntimeError("Frame overlap detected between train and valid masks.")
    if int(train_mask.sum() + valid_mask.sum()) != n:
        raise RuntimeError("Some frames were not assigned to train or valid split.")

    clean_mask = task_labels == 0
    obstacle_scenario_mask = task_labels == 1
    visible_mask = obstacle_visible

    train_data = _select_by_mask(data, train_mask)
    valid_data = _select_by_mask(data, valid_mask)

    clean_pref_train = _select_by_mask(data, train_mask & clean_mask)
    clean_pref_valid = _select_by_mask(data, valid_mask & clean_mask)

    obs_vis_pref_train = _select_by_mask(
        data,
        train_mask & obstacle_scenario_mask & visible_mask,
    )
    obs_vis_pref_valid = _select_by_mask(
        data,
        valid_mask & obstacle_scenario_mask & visible_mask,
    )

    _write_h5(
        output_dir / "world_model_train_v5.h5",
        train_data,
        _add_common_attrs(input_attrs, "world_model_train_v5", len(train_data["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )
    _write_h5(
        output_dir / "world_model_valid_v5.h5",
        valid_data,
        _add_common_attrs(input_attrs, "world_model_valid_v5", len(valid_data["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )

    _write_h5(
        output_dir / "clean_preference_train_v5.h5",
        clean_pref_train,
        _add_common_attrs(input_attrs, "clean_preference_train_v5", len(clean_pref_train["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )
    _write_h5(
        output_dir / "clean_preference_valid_v5.h5",
        clean_pref_valid,
        _add_common_attrs(input_attrs, "clean_preference_valid_v5", len(clean_pref_valid["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )
    _write_h5(
        output_dir / "obstacle_visible_preference_train_v5.h5",
        obs_vis_pref_train,
        _add_common_attrs(input_attrs, "obstacle_visible_preference_train_v5", len(obs_vis_pref_train["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )
    _write_h5(
        output_dir / "obstacle_visible_preference_valid_v5.h5",
        obs_vis_pref_valid,
        _add_common_attrs(input_attrs, "obstacle_visible_preference_valid_v5", len(obs_vis_pref_valid["episode_ids"]), args.train_ratio, args.seed, args.seq_len),
        compression=args.compression,
    )

    _write_grouped_h5(
        output_dir / "pp_gap_eval_v5.h5",
        {
            "clean": clean_pref_valid,
            "obstacle_visible": obs_vis_pref_valid,
        },
        {
            "collection_type": "pp_gap_eval_v5",
            "clean_frames": int(len(clean_pref_valid["episode_ids"])),
            "obstacle_visible_frames": int(len(obs_vis_pref_valid["episode_ids"])),
            "train_ratio": float(args.train_ratio),
            "split_seed": int(args.seed),
            "source_input": str(input_path),
        },
        compression=args.compression,
    )

    with h5py.File(output_dir / "split_metadata_v5.h5", "w") as f:
        f.create_dataset("train_episode_ids", data=train_eps)
        f.create_dataset("valid_episode_ids", data=valid_eps)
        f.create_dataset("train_frame_mask", data=train_mask.astype(bool))
        f.create_dataset("valid_frame_mask", data=valid_mask.astype(bool))
        f.attrs["source_input"] = str(input_path)
        f.attrs["train_ratio"] = float(args.train_ratio)
        f.attrs["split_seed"] = int(args.seed)
        f.attrs["total_frames"] = int(n)
        f.attrs["train_frames"] = int(train_mask.sum())
        f.attrs["valid_frames"] = int(valid_mask.sum())
        f.attrs["train_episodes"] = int(len(train_eps))
        f.attrs["valid_episodes"] = int(len(valid_eps))
        f.attrs["train_visible_frames"] = int((train_mask & visible_mask).sum())
        f.attrs["valid_visible_frames"] = int((valid_mask & visible_mask).sum())

    print("=" * 100)
    print("Split summary")
    _print_file_summary("world_model_train_v5.h5", train_data, args.seq_len)
    _print_file_summary("world_model_valid_v5.h5", valid_data, args.seq_len)
    _print_file_summary("clean_preference_train_v5.h5", clean_pref_train, args.seq_len)
    _print_file_summary("clean_preference_valid_v5.h5", clean_pref_valid, args.seq_len)
    _print_file_summary("obstacle_visible_preference_train_v5.h5", obs_vis_pref_train, args.seq_len)
    _print_file_summary("obstacle_visible_preference_valid_v5.h5", obs_vis_pref_valid, args.seq_len)

    print("=" * 100)
    print(f"Wrote split files to: {output_dir}")
    print("Use for world-model training:")
    print(f"  --data {output_dir / 'world_model_train_v5.h5'}")
    print("Use for held-out world-model/PP-Gap evaluation:")
    print(f"  {output_dir / 'world_model_valid_v5.h5'}")
    print(f"  {output_dir / 'pp_gap_eval_v5.h5'}")


if __name__ == "__main__":
    main()
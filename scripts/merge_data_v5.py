"""Merge PAActInf HDF5 datasets with v5 obstacle metadata support.

Why this script exists
----------------------
The legacy merge_data.py concatenates keys that already exist in each input
file. That is unsafe for v5 because old clean datasets do not contain keys such
as obstacle_visible, obstacle_distance, visible_obstacle_id, or obstacle_bbox.

This script creates a unified world-model training HDF5 where every per-frame
key has the same length as images/states/actions.

Expected usage
--------------
uv run python scripts/merge_data_v5.py \
    --inputs \
        data/expert_data_town04.h5:0 \
        data/expert_clean_town06_v5.h5:0 \
        data/expert_obstacle_v5.h5:1 \
    --output data/world_model_train_v5.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_input_spec(spec: str) -> tuple[str, int | None]:
    """Parse 'path' or 'path:task_label'.

    Examples:
        data/a.h5      -> ("data/a.h5", None)
        data/a.h5:0    -> ("data/a.h5", 0)
        data/a.h5:1    -> ("data/a.h5", 1)
    """
    if ":" not in spec:
        return spec, None

    path, label = spec.rsplit(":", 1)
    if label not in {"0", "1"}:
        raise ValueError(
            f"Invalid input spec '{spec}'. Use path or path:0/path:1."
        )
    return path, int(label)


def read_images(f: h5py.File) -> np.ndarray:
    """Read images from 'images' or legacy 'frames' key.

    Output is always float32, shape (T, C, H, W).
    """
    if "images" in f:
        images = np.asarray(f["images"][:])
    elif "frames" in f:
        images = np.asarray(f["frames"][:])
    else:
        raise KeyError("Input file has neither 'images' nor 'frames' key.")

    # Convert HWC to CHW if needed.
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2)

    images = images.astype(np.float32)

    # Convert uint8-like [0,255] to normalized [-0.5,0.5] if needed.
    if images.max() > 2.0:
        images = images / 255.0 - 0.5

    return images


def read_required(f: h5py.File, key: str) -> np.ndarray:
    if key not in f:
        raise KeyError(f"Required key missing: {key}")
    return np.asarray(f[key][:])


def get_1d(
    f: h5py.File,
    key: str,
    n: int,
    dtype,
    default,
) -> np.ndarray:
    if key in f:
        arr = np.asarray(f[key][:])
        if len(arr) != n:
            raise ValueError(f"Length mismatch for {key}: {len(arr)} != {n}")
        return arr.astype(dtype)
    return np.full((n,), default, dtype=dtype)


def get_2d(
    f: h5py.File,
    key: str,
    n: int,
    dim: int,
    dtype,
    default,
) -> np.ndarray:
    if key in f:
        arr = np.asarray(f[key][:])
        if arr.shape[0] != n:
            raise ValueError(f"Length mismatch for {key}: {arr.shape[0]} != {n}")
        return arr.astype(dtype)

    arr = np.full((n, dim), default, dtype=dtype)
    return arr


def remap_episode_ids(episode_ids: np.ndarray, next_episode_id: int) -> tuple[np.ndarray, int]:
    """Map source episode ids to globally unique sequential ids."""
    mapping: dict[int, int] = {}
    out = np.empty_like(episode_ids, dtype=np.int64)

    for i, ep in enumerate(episode_ids):
        ep_int = int(ep)
        if ep_int not in mapping:
            mapping[ep_int] = next_episode_id
            next_episode_id += 1
        out[i] = mapping[ep_int]

    return out, next_episode_id


def pad_states_to_max_dim(states_list: list[np.ndarray], pad_value: float = 1.0) -> list[np.ndarray]:
    max_dim = max(s.shape[1] for s in states_list)
    padded = []

    for states in states_list:
        if states.shape[1] == max_dim:
            padded.append(states.astype(np.float32))
            continue

        pad_width = max_dim - states.shape[1]
        pad = np.full((states.shape[0], pad_width), pad_value, dtype=np.float32)
        padded.append(np.concatenate([states.astype(np.float32), pad], axis=1))

    return padded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PAActInf HDF5 datasets with v5 obstacle metadata defaults.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input HDF5 files. Use path:0 or path:1 to override task_labels.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_images: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_expert_actions: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []

    all_lateral_devs: list[np.ndarray] = []
    all_lane_ids: list[np.ndarray] = []
    all_noise_sigmas: list[np.ndarray] = []
    all_success_flags: list[np.ndarray] = []
    all_task_labels: list[np.ndarray] = []

    all_obstacle_visible: list[np.ndarray] = []
    all_obstacle_distance: list[np.ndarray] = []
    all_obstacle_in_front: list[np.ndarray] = []
    all_obstacle_lateral: list[np.ndarray] = []
    all_nearest_obstacle_id: list[np.ndarray] = []
    all_visible_obstacle_id: list[np.ndarray] = []
    all_obstacle_bbox: list[np.ndarray] = []
    all_obstacle_bbox_area: list[np.ndarray] = []

    all_ego_x: list[np.ndarray] = []
    all_ego_y: list[np.ndarray] = []
    all_ego_yaw: list[np.ndarray] = []
    all_obstacle_x: list[np.ndarray] = []
    all_obstacle_y: list[np.ndarray] = []
    all_source_ids: list[np.ndarray] = []

    next_episode_id = 0
    source_attrs: dict[str, str | int] = {}

    for source_id, spec in enumerate(args.inputs):
        path, task_override = parse_input_spec(spec)
        print("=" * 100)
        print(f"[{source_id}] Loading {path}")
        print(f"task_label_override: {task_override}")

        with h5py.File(path, "r") as f:
            images = read_images(f)
            states = read_required(f, "states").astype(np.float32)
            actions = read_required(f, "actions").astype(np.float32)
            n = images.shape[0]

            if states.shape[0] != n:
                raise ValueError(f"{path}: states length {states.shape[0]} != images length {n}")

            # Some datasets may store T-1 actions. Pad to T.
            if actions.shape[0] == n - 1:
                pad = np.zeros((1, actions.shape[1]), dtype=np.float32)
                actions = np.concatenate([actions, pad], axis=0)
            if actions.shape[0] != n:
                raise ValueError(f"{path}: actions length {actions.shape[0]} != images length {n}")

            action_dim = actions.shape[1]

            if "episode_ids" in f:
                episode_ids_raw = np.asarray(f["episode_ids"][:], dtype=np.int64)
            else:
                episode_ids_raw = np.zeros((n,), dtype=np.int64)

            if len(episode_ids_raw) != n:
                raise ValueError(f"{path}: episode_ids length {len(episode_ids_raw)} != {n}")

            episode_ids, next_episode_id = remap_episode_ids(
                episode_ids_raw,
                next_episode_id,
            )

            if "expert_actions" in f:
                expert_actions = np.asarray(f["expert_actions"][:], dtype=np.float32)
                if expert_actions.shape[0] != n:
                    raise ValueError(f"{path}: expert_actions length mismatch.")
            else:
                expert_actions = actions.copy()

            if task_override is not None:
                task_labels = np.full((n,), task_override, dtype=np.int8)
            elif "task_labels" in f:
                task_labels = np.asarray(f["task_labels"][:], dtype=np.int8)
                if len(task_labels) != n:
                    raise ValueError(f"{path}: task_labels length mismatch.")
            else:
                task_labels = np.zeros((n,), dtype=np.int8)

            all_images.append(images)
            all_states.append(states)
            all_actions.append(actions)
            all_expert_actions.append(expert_actions)
            all_episode_ids.append(episode_ids)

            all_lateral_devs.append(get_1d(f, "lateral_devs", n, np.float32, 0.0))
            all_lane_ids.append(get_1d(f, "lane_ids", n, np.int32, -1))
            all_noise_sigmas.append(get_2d(f, "noise_sigmas", n, action_dim, np.float32, 0.0))
            all_success_flags.append(get_1d(f, "success_flags", n, bool, True))
            all_task_labels.append(task_labels)

            all_obstacle_visible.append(get_1d(f, "obstacle_visible", n, bool, False))
            all_obstacle_distance.append(get_1d(f, "obstacle_distance", n, np.float32, np.inf))
            all_obstacle_in_front.append(get_1d(f, "obstacle_in_front", n, bool, False))
            all_obstacle_lateral.append(get_1d(f, "obstacle_lateral", n, np.float32, np.nan))
            all_nearest_obstacle_id.append(get_1d(f, "nearest_obstacle_id", n, np.int32, -1))
            all_visible_obstacle_id.append(get_1d(f, "visible_obstacle_id", n, np.int32, -1))
            all_obstacle_bbox.append(get_2d(f, "obstacle_bbox", n, 4, np.float32, np.nan))
            all_obstacle_bbox_area.append(get_1d(f, "obstacle_bbox_area", n, np.float32, 0.0))

            all_ego_x.append(get_1d(f, "ego_x", n, np.float32, np.nan))
            all_ego_y.append(get_1d(f, "ego_y", n, np.float32, np.nan))
            all_ego_yaw.append(get_1d(f, "ego_yaw", n, np.float32, np.nan))
            all_obstacle_x.append(get_1d(f, "obstacle_x", n, np.float32, np.nan))
            all_obstacle_y.append(get_1d(f, "obstacle_y", n, np.float32, np.nan))
            all_source_ids.append(np.full((n,), source_id, dtype=np.int32))

            source_attrs[f"source_{source_id}_path"] = path
            source_attrs[f"source_{source_id}_task_label_override"] = (
                -1 if task_override is None else task_override
            )
            source_attrs[f"source_{source_id}_frames"] = int(n)

            print(f"frames: {n}")
            print(f"states: {states.shape}")
            print(f"actions: {actions.shape}")
            print(f"visible frames: {int(all_obstacle_visible[-1].sum())}")

    # State dims can differ across old/new files.
    all_states = pad_states_to_max_dim(all_states, pad_value=1.0)

    merged = {
        "images": np.concatenate(all_images, axis=0).astype(np.float32),
        "states": np.concatenate(all_states, axis=0).astype(np.float32),
        "actions": np.concatenate(all_actions, axis=0).astype(np.float32),
        "expert_actions": np.concatenate(all_expert_actions, axis=0).astype(np.float32),
        "episode_ids": np.concatenate(all_episode_ids, axis=0).astype(np.int64),

        "lateral_devs": np.concatenate(all_lateral_devs, axis=0).astype(np.float32),
        "lane_ids": np.concatenate(all_lane_ids, axis=0).astype(np.int32),
        "noise_sigmas": np.concatenate(all_noise_sigmas, axis=0).astype(np.float32),
        "success_flags": np.concatenate(all_success_flags, axis=0).astype(bool),
        "task_labels": np.concatenate(all_task_labels, axis=0).astype(np.int8),

        "obstacle_visible": np.concatenate(all_obstacle_visible, axis=0).astype(bool),
        "obstacle_distance": np.concatenate(all_obstacle_distance, axis=0).astype(np.float32),
        "obstacle_in_front": np.concatenate(all_obstacle_in_front, axis=0).astype(bool),
        "obstacle_lateral": np.concatenate(all_obstacle_lateral, axis=0).astype(np.float32),
        "nearest_obstacle_id": np.concatenate(all_nearest_obstacle_id, axis=0).astype(np.int32),
        "visible_obstacle_id": np.concatenate(all_visible_obstacle_id, axis=0).astype(np.int32),
        "obstacle_bbox": np.concatenate(all_obstacle_bbox, axis=0).astype(np.float32),
        "obstacle_bbox_area": np.concatenate(all_obstacle_bbox_area, axis=0).astype(np.float32),

        "ego_x": np.concatenate(all_ego_x, axis=0).astype(np.float32),
        "ego_y": np.concatenate(all_ego_y, axis=0).astype(np.float32),
        "ego_yaw": np.concatenate(all_ego_yaw, axis=0).astype(np.float32),
        "obstacle_x": np.concatenate(all_obstacle_x, axis=0).astype(np.float32),
        "obstacle_y": np.concatenate(all_obstacle_y, axis=0).astype(np.float32),
        "source_ids": np.concatenate(all_source_ids, axis=0).astype(np.int32),
    }

    total = merged["images"].shape[0]
    print("=" * 100)
    print(f"Writing {args.output}")
    print(f"total frames: {total}")
    print(f"total episodes: {next_episode_id}")
    print(f"visible frames: {int(merged['obstacle_visible'].sum())}")
    print(f"state dim: {merged['states'].shape[1]}")
    print(f"action dim: {merged['actions'].shape[1]}")

    # Length consistency check.
    for key, value in merged.items():
        if value.shape[0] != total:
            raise ValueError(f"Merged key length mismatch: {key} {value.shape[0]} != {total}")

    with h5py.File(args.output, "w") as f:
        for key, value in merged.items():
            f.create_dataset(key, data=value)

        f.attrs["collection_type"] = "merged_world_model_train_v5"
        f.attrs["total_frames"] = int(total)
        f.attrs["total_episodes"] = int(next_episode_id)
        f.attrs["num_sources"] = int(len(args.inputs))
        f.attrs["state_dim"] = int(merged["states"].shape[1])
        f.attrs["action_dim"] = int(merged["actions"].shape[1])
        f.attrs["visible_frames"] = int(merged["obstacle_visible"].sum())

        for key, value in source_attrs.items():
            f.attrs[key] = value

    print("Done.")


if __name__ == "__main__":
    main()
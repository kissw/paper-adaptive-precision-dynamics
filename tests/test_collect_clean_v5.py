"""Unit tests for collect_clean_data_v5.py — schema consistency checks.

Tests verify:
  1. The full set of HDF5 keys produced by clean v5 matches obstacle v5.
  2. obstacle_visible is always False in a synthetic clean-data HDF5.
  3. obstacle_distance is +inf for all clean frames.
  4. nearest_obstacle_id is -1 for all clean frames.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ─────────────────────────────────────────────────────────────────────────────
# Expected key sets
# ─────────────────────────────────────────────────────────────────────────────

V5_BASE_KEYS = {
    "images", "states", "actions", "expert_actions", "episode_ids",
    "lateral_devs", "lane_ids", "noise_sigmas", "success_flags", "task_labels",
}

V5_OBSTACLE_KEYS = {
    "obstacle_visible", "obstacle_distance", "obstacle_in_front",
    "obstacle_lateral", "nearest_obstacle_id",
    "obstacle_bbox", "obstacle_bbox_area",
    "ego_x", "ego_y", "ego_yaw",
    "obstacle_x", "obstacle_y",
}

V5_ALL_KEYS = V5_BASE_KEYS | V5_OBSTACLE_KEYS


def _make_synthetic_clean_h5(path: Path, n: int = 20) -> None:
    """Write a minimal synthetic clean-v5 HDF5 to verify schema expectations."""
    with h5py.File(path, "w") as f:
        # Base keys
        f.create_dataset("images", data=np.zeros((n, 3, 64, 64), dtype=np.float32))
        f.create_dataset("states", data=np.zeros((n, 4), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros((n, 2), dtype=np.float32))
        f.create_dataset("expert_actions", data=np.zeros((n, 2), dtype=np.float32))
        f.create_dataset("episode_ids", data=np.arange(n, dtype=np.int64))
        f.create_dataset("lateral_devs", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("lane_ids", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("noise_sigmas", data=np.zeros((n, 2), dtype=np.float32))
        f.create_dataset("success_flags", data=np.ones(n, dtype=bool))
        f.create_dataset("task_labels", data=np.zeros(n, dtype=np.int8))
        # v5 obstacle keys — clean placeholders
        f.create_dataset("obstacle_visible", data=np.zeros(n, dtype=bool))
        f.create_dataset("obstacle_distance", data=np.full(n, np.inf, dtype=np.float32))
        f.create_dataset("obstacle_in_front", data=np.zeros(n, dtype=bool))
        f.create_dataset("obstacle_lateral", data=np.full(n, np.nan, dtype=np.float32))
        f.create_dataset("nearest_obstacle_id", data=np.full(n, -1, dtype=np.int32))
        f.create_dataset("obstacle_bbox", data=np.full((n, 4), np.nan, dtype=np.float32))
        f.create_dataset("obstacle_bbox_area", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_x", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_y", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("obstacle_x", data=np.full(n, np.nan, dtype=np.float32))
        f.create_dataset("obstacle_y", data=np.full(n, np.nan, dtype=np.float32))
        f.attrs["collection_type"] = "clean_normal_driving_v5_town06_same_route"
        f.attrs["obstacle_count"] = 0


def _make_synthetic_obstacle_h5(path: Path, n: int = 20) -> None:
    """Write a minimal synthetic obstacle-v5 HDF5 to compare key sets."""
    with h5py.File(path, "w") as f:
        for key in V5_BASE_KEYS:
            if key in ("images",):
                f.create_dataset(key, data=np.zeros((n, 3, 64, 64), dtype=np.float32))
            elif key in ("success_flags",):
                f.create_dataset(key, data=np.ones(n, dtype=bool))
            elif key in ("task_labels",):
                f.create_dataset(key, data=np.ones(n, dtype=np.int8))
            elif key in ("episode_ids",):
                f.create_dataset(key, data=np.zeros(n, dtype=np.int64))
            elif key in ("noise_sigmas",):
                f.create_dataset(key, data=np.zeros((n, 2), dtype=np.float32))
            elif key in ("lane_ids",):
                f.create_dataset(key, data=np.zeros(n, dtype=np.int32))
            elif key in ("states", "actions", "expert_actions"):
                f.create_dataset(key, data=np.zeros((n, 2), dtype=np.float32))
            else:
                f.create_dataset(key, data=np.zeros(n, dtype=np.float32))
        f.create_dataset("obstacle_visible", data=np.zeros(n, dtype=bool))
        f.create_dataset("obstacle_distance", data=np.full(n, 99.0, dtype=np.float32))
        f.create_dataset("obstacle_in_front", data=np.zeros(n, dtype=bool))
        f.create_dataset("obstacle_lateral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("nearest_obstacle_id", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("obstacle_bbox", data=np.full((n, 4), np.nan, dtype=np.float32))
        f.create_dataset("obstacle_bbox_area", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_x", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_y", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("obstacle_x", data=np.full(n, np.nan, dtype=np.float32))
        f.create_dataset("obstacle_y", data=np.full(n, np.nan, dtype=np.float32))
        f.attrs["collection_type"] = "waypoint_obstacle_avoidance_v5_visible_labeled"


class TestCleanV5Schema:

    @pytest.fixture()
    def clean_h5(self, tmp_path):
        path = tmp_path / "clean_v5.h5"
        _make_synthetic_clean_h5(path)
        return path

    @pytest.fixture()
    def obstacle_h5(self, tmp_path):
        path = tmp_path / "obstacle_v5.h5"
        _make_synthetic_obstacle_h5(path)
        return path

    def test_clean_has_all_v5_keys(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            keys = set(f.keys())
        assert V5_ALL_KEYS.issubset(keys), (
            f"Missing keys: {V5_ALL_KEYS - keys}"
        )

    def test_key_set_matches_obstacle_v5(self, clean_h5, obstacle_h5):
        with h5py.File(clean_h5, "r") as f:
            clean_keys = set(f.keys())
        with h5py.File(obstacle_h5, "r") as f:
            obstacle_keys = set(f.keys())
        assert clean_keys == obstacle_keys, (
            f"Key mismatch — only in clean: {clean_keys - obstacle_keys}, "
            f"only in obstacle: {obstacle_keys - clean_keys}"
        )

    def test_obstacle_visible_is_all_false(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            visible = f["obstacle_visible"][:]
        assert not np.any(visible), "obstacle_visible must be all False for clean data"

    def test_obstacle_distance_is_inf(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            dist = f["obstacle_distance"][:]
        assert np.all(np.isinf(dist)), "obstacle_distance must be +inf for clean data"

    def test_nearest_obstacle_id_is_minus_one(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            ids = f["nearest_obstacle_id"][:]
        assert np.all(ids == -1), "nearest_obstacle_id must be -1 for clean data"

    def test_collection_type_attr(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            ct = f.attrs["collection_type"]
        assert ct == "clean_normal_driving_v5_town06_same_route"

    def test_obstacle_count_attr_is_zero(self, clean_h5):
        with h5py.File(clean_h5, "r") as f:
            count = f.attrs["obstacle_count"]
        assert count == 0

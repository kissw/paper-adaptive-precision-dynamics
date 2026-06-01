"""Unit tests for build_v5_subsets.py."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_v5_subsets import (
    episode_split,
    load_h5,
    merge_datasets,
    select_by_episode,
    select_by_mask,
    write_h5,
    write_h5_groups,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers to generate synthetic v5 HDF5 data dicts
# ─────────────────────────────────────────────────────────────────────────────

def _make_data(n_episodes: int = 5, frames_per_ep: int = 10,
               has_visible: bool = False) -> dict:
    """Synthetic v5 data dict (no file I/O)."""
    T = n_episodes * frames_per_ep
    episode_ids = np.repeat(np.arange(n_episodes, dtype=np.int64), frames_per_ep)
    visible = np.zeros(T, dtype=bool)
    if has_visible:
        # Mark middle frames of each episode as visible
        for ep in range(n_episodes):
            start = ep * frames_per_ep
            mid = start + frames_per_ep // 4
            visible[mid: mid + frames_per_ep // 2] = True
    return {
        "images":              np.zeros((T, 3, 64, 64), dtype=np.float32),
        "states":              np.zeros((T, 4), dtype=np.float32),
        "actions":             np.zeros((T, 2), dtype=np.float32),
        "expert_actions":      np.zeros((T, 2), dtype=np.float32),
        "episode_ids":         episode_ids,
        "lateral_devs":        np.zeros(T, dtype=np.float32),
        "lane_ids":            np.zeros(T, dtype=np.int32),
        "noise_sigmas":        np.zeros((T, 2), dtype=np.float32),
        "success_flags":       np.ones(T, dtype=bool),
        "task_labels":         np.zeros(T, dtype=np.int8),
        "obstacle_visible":    visible,
        "obstacle_distance":   np.full(T, np.inf, dtype=np.float32),
        "obstacle_in_front":   np.zeros(T, dtype=bool),
        "obstacle_lateral":    np.full(T, np.nan, dtype=np.float32),
        "nearest_obstacle_id": np.full(T, -1, dtype=np.int32),
        "obstacle_bbox":       np.full((T, 4), np.nan, dtype=np.float32),
        "obstacle_bbox_area":  np.zeros(T, dtype=np.float32),
        "ego_x":               np.zeros(T, dtype=np.float32),
        "ego_y":               np.zeros(T, dtype=np.float32),
        "ego_yaw":             np.zeros(T, dtype=np.float32),
        "obstacle_x":          np.full(T, np.nan, dtype=np.float32),
        "obstacle_y":          np.full(T, np.nan, dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# episode_split
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeSplit:

    def test_split_ratio(self):
        episode_ids = np.arange(10)
        ids_arr = np.repeat(episode_ids, 5)
        train_ids, eval_ids = episode_split(ids_arr, 0.7, seed=42)
        assert len(train_ids) == 7
        assert len(eval_ids) == 3

    def test_no_overlap(self):
        ids_arr = np.repeat(np.arange(20), 5)
        train_ids, eval_ids = episode_split(ids_arr, 0.7, seed=0)
        assert train_ids.isdisjoint(eval_ids)

    def test_all_episodes_covered(self):
        unique_eps = np.arange(10)
        ids_arr = np.repeat(unique_eps, 5)
        train_ids, eval_ids = episode_split(ids_arr, 0.7, seed=1)
        assert train_ids | eval_ids == set(unique_eps.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# select_by_episode
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectByEpisode:

    def test_correct_frames_selected(self):
        data = _make_data(n_episodes=4, frames_per_ep=5)
        subset = select_by_episode(data, {0, 2})
        assert len(subset["episode_ids"]) == 10  # 2 eps × 5 frames
        assert set(subset["episode_ids"].tolist()) == {0, 2}

    def test_all_arrays_same_length(self):
        data = _make_data(n_episodes=4, frames_per_ep=5)
        subset = select_by_episode(data, {1, 3})
        n = len(subset["episode_ids"])
        for k, v in subset.items():
            if isinstance(v, np.ndarray):
                assert len(v) == n, f"Key {k} has length {len(v)}, expected {n}"


# ─────────────────────────────────────────────────────────────────────────────
# merge_datasets
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeDatasets:

    def test_episode_id_no_collision_after_offset(self):
        obs = _make_data(n_episodes=5, frames_per_ep=4)
        cln = _make_data(n_episodes=3, frames_per_ep=4)
        offset = int(obs["episode_ids"].max()) + 1
        merged = merge_datasets(obs, cln, b_episode_id_offset=offset)
        unique_ids = np.unique(merged["episode_ids"])
        # All IDs must be unique (no duplicates after offset)
        assert len(unique_ids) == 5 + 3

    def test_total_frame_count(self):
        obs = _make_data(n_episodes=5, frames_per_ep=4)
        cln = _make_data(n_episodes=3, frames_per_ep=4)
        offset = int(obs["episode_ids"].max()) + 1
        merged = merge_datasets(obs, cln, b_episode_id_offset=offset)
        assert len(merged["episode_ids"]) == (5 + 3) * 4

    def test_offset_applied_to_clean_ids_only(self):
        obs = _make_data(n_episodes=3, frames_per_ep=2)  # ids 0,1,2
        cln = _make_data(n_episodes=3, frames_per_ep=2)  # ids 0,1,2
        offset = 10
        merged = merge_datasets(obs, cln, b_episode_id_offset=offset)
        # obs ids should remain 0,1,2; cln ids should be 10,11,12
        ids = set(merged["episode_ids"].tolist())
        assert {0, 1, 2, 10, 11, 12} == ids


# ─────────────────────────────────────────────────────────────────────────────
# select_by_mask (visible frames)
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectByMask:

    def test_visible_only(self):
        data = _make_data(n_episodes=4, frames_per_ep=10, has_visible=True)
        mask = data["obstacle_visible"].astype(bool)
        subset = select_by_mask(data, mask)
        assert np.all(subset["obstacle_visible"])

    def test_subset_length_matches_mask(self):
        data = _make_data(n_episodes=4, frames_per_ep=10, has_visible=True)
        mask = data["obstacle_visible"].astype(bool)
        subset = select_by_mask(data, mask)
        assert len(subset["episode_ids"]) == int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full build pipeline on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPipeline:

    @pytest.fixture()
    def paths(self, tmp_path):
        obs_path = str(tmp_path / "obstacle.h5")
        cln_path = str(tmp_path / "clean.h5")

        obs_data = _make_data(n_episodes=10, frames_per_ep=8, has_visible=True)
        cln_data = _make_data(n_episodes=10, frames_per_ep=8)

        write_h5(obs_path, obs_data, {"collection_type": "obstacle"})
        write_h5(cln_path, cln_data, {"collection_type": "clean"})
        return obs_path, cln_path, tmp_path

    def test_train_eval_episode_ids_disjoint(self, paths):
        obs_path, cln_path, tmp_path = paths
        obs = load_h5(obs_path)
        cln = load_h5(cln_path)

        train_eps, eval_eps = episode_split(obs["episode_ids"], 0.7, seed=42)
        obs_train = select_by_episode(obs, train_eps)
        obs_eval  = select_by_episode(obs, eval_eps)

        train_ids = set(obs_train["episode_ids"].tolist())
        eval_ids  = set(obs_eval["episode_ids"].tolist())
        assert train_ids.isdisjoint(eval_ids)

    def test_obstacle_visible_train_all_visible(self, paths):
        obs_path, _, tmp_path = paths
        obs = load_h5(obs_path)
        train_eps, _ = episode_split(obs["episode_ids"], 0.7, seed=42)
        obs_train = select_by_episode(obs, train_eps)
        mask = obs_train["obstacle_visible"].astype(bool)
        vis_train = select_by_mask(obs_train, mask)
        assert np.all(vis_train["obstacle_visible"])

    def test_clean_preference_all_not_visible(self, paths):
        _, cln_path, tmp_path = paths
        cln = load_h5(cln_path)
        train_eps, _ = episode_split(cln["episode_ids"], 0.7, seed=42)
        cln_train = select_by_episode(cln, train_eps)
        assert not np.any(cln_train["obstacle_visible"])

    def test_write_h5_groups_readable(self, paths, tmp_path):
        obs_path, cln_path, _ = paths
        obs = load_h5(obs_path)
        cln = load_h5(cln_path)

        _, obs_eval_eps = episode_split(obs["episode_ids"], 0.7, seed=42)
        _, cln_eval_eps = episode_split(cln["episode_ids"], 0.7, seed=42)
        obs_eval = select_by_episode(obs, obs_eval_eps)
        cln_eval = select_by_episode(cln, cln_eval_eps)

        out_path = str(tmp_path / "eval.h5")
        write_h5_groups(
            out_path,
            {"clean": cln_eval, "obstacle_visible": obs_eval},
            {"collection_type": "pp_gap_eval_v5"},
        )

        with h5py.File(out_path, "r") as f:
            assert "clean" in f
            assert "obstacle_visible" in f
            assert f.attrs["collection_type"] == "pp_gap_eval_v5"
            assert "episode_ids" in f["clean"]
            assert "episode_ids" in f["obstacle_visible"]

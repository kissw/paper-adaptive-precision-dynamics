"""Tests for scripts/merge_wm_train_v6.py."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from merge_wm_train_v6 import (
    REQUIRED_KEYS,
    check_schema_compatibility,
    merge_datasets,
    sample_cf_frames,
)


def _make_h5(path: Path, T: int, n_episodes: int, extra_keys: list[str] | None = None) -> None:
    """Write a minimal synthetic HDF5 with REQUIRED_KEYS + obstacle metadata."""
    assert T % n_episodes == 0, "T must be divisible by n_episodes"
    frames_per_ep = T // n_episodes
    ep_ids = np.repeat(np.arange(n_episodes), frames_per_ep).astype(np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=np.zeros((T, 3, 64, 64), dtype=np.float32))
        f.create_dataset("states", data=np.zeros((T, 4), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros((T, 2), dtype=np.float32))
        f.create_dataset("episode_ids", data=ep_ids)
        f.create_dataset("obstacle_visible", data=np.zeros(T, dtype=bool))
        f.create_dataset("obstacle_distance", data=np.full(T, 100.0, dtype=np.float32))
        f.create_dataset("ego_x", data=np.zeros(T, dtype=np.float32))
        f.create_dataset("ego_y", data=np.zeros(T, dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.zeros(T, dtype=np.float32))
        if extra_keys:
            for k in extra_keys:
                f.create_dataset(k, data=np.zeros(T, dtype=np.float32))


class TestSchemaCheck:
    def test_passes_when_all_required_keys_present(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 10, 2)
        check_schema_compatibility(exp, cf)  # should not raise

    def test_raises_when_expert_missing_key(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        # Write expert without 'actions'
        with h5py.File(exp, "w") as f:
            f.create_dataset("images", data=np.zeros((4, 3, 64, 64), dtype=np.float32))
            f.create_dataset("states", data=np.zeros((4, 4), dtype=np.float32))
            f.create_dataset("episode_ids", data=np.zeros(4, dtype=np.int64))
        _make_h5(cf, 4, 1)
        with pytest.raises(ValueError, match="expert.*missing.*actions"):
            check_schema_compatibility(exp, cf)

    def test_raises_when_cf_missing_key(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 4, 1)
        # Write CF without 'episode_ids'
        with h5py.File(cf, "w") as f:
            f.create_dataset("images", data=np.zeros((4, 3, 64, 64), dtype=np.float32))
            f.create_dataset("states", data=np.zeros((4, 4), dtype=np.float32))
            f.create_dataset("actions", data=np.zeros((4, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="cf.*missing.*episode_ids"):
            check_schema_compatibility(exp, cf)


class TestEpisodeIdOffset:
    def test_no_collision_after_merge(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"

        _make_h5(exp, 10, 2)   # expert episode_ids: 0, 1
        _make_h5(cf, 12, 3)    # cf episode_ids: 0, 1, 2 → offset to 2, 3, 4

        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)

        with h5py.File(out, "r") as f:
            ep_ids = f["episode_ids"][:]

        expert_ids = set(range(2))           # 0, 1
        cf_ids_after_offset = set(range(2, 5))  # 2, 3, 4
        unique_ids = set(ep_ids.tolist())
        assert expert_ids.issubset(unique_ids)
        assert cf_ids_after_offset.issubset(unique_ids)
        assert expert_ids.isdisjoint(cf_ids_after_offset)

    def test_episode_ids_are_contiguous_integers(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 8, 2)
        _make_h5(cf, 6, 3)
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            ep_ids = sorted(set(f["episode_ids"][:].tolist()))
        assert ep_ids == list(range(len(ep_ids)))


class TestCfRatioSampling:
    def test_cf_ratio_zero_returns_only_expert(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 20, 4)
        summary = merge_datasets(exp, cf, out, cf_ratio=0.0, seed=0)
        assert summary["n_cf_frames_sampled"] == 0
        assert summary["total_frames"] == 10

    def test_cf_ratio_one_includes_all_cf(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 20, 4)
        summary = merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        assert summary["n_cf_frames_sampled"] == 20
        assert summary["total_frames"] == 30

    def test_cf_ratio_half_approximately_correct(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 100, 10)
        summary = merge_datasets(exp, cf, out, cf_ratio=0.5, seed=0)
        assert summary["n_cf_frames_sampled"] == 50
        assert summary["total_frames"] == 60

    def test_same_seed_produces_same_sample(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 4, 1)
        _make_h5(cf, 20, 4)
        out_a = tmp_path / "merged_a.h5"
        out_b = tmp_path / "merged_b.h5"
        merge_datasets(exp, cf, out_a, cf_ratio=0.5, seed=42)
        merge_datasets(exp, cf, out_b, cf_ratio=0.5, seed=42)
        with h5py.File(out_a, "r") as fa, h5py.File(out_b, "r") as fb:
            np.testing.assert_array_equal(fa["episode_ids"][:], fb["episode_ids"][:])

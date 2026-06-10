"""Tests for scripts/merge_wm_train_v6.py."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from merge_wm_train_v6 import (
    CF_FLAG_KEYS,
    REQUIRED_KEYS,
    check_schema_compatibility,
    merge_datasets,
    sample_cf_frames,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_h5(
    path: Path,
    T: int,
    n_episodes: int,
    *,
    state_dim: int = 4,
    action_dim: int = 2,
    image_hw: int = 64,
    cf_flags: bool = False,
    frames_per_ep: int | None = None,
) -> None:
    """Write a minimal synthetic HDF5.

    If cf_flags=True, also writes CF identification keys so we can verify
    flag propagation in merge tests.
    If frames_per_ep is provided, T is ignored and total frames = n_episodes * frames_per_ep.
    """
    if frames_per_ep is not None:
        T = n_episodes * frames_per_ep
    else:
        assert T % n_episodes == 0, "T must be divisible by n_episodes"
        frames_per_ep = T // n_episodes

    ep_ids = np.repeat(np.arange(n_episodes), frames_per_ep).astype(np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=np.zeros((T, 3, image_hw, image_hw), dtype=np.float32))
        f.create_dataset("states", data=np.zeros((T, state_dim), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros((T, action_dim), dtype=np.float32))
        f.create_dataset("episode_ids", data=ep_ids)
        f.create_dataset("obstacle_visible", data=np.zeros(T, dtype=bool))
        f.create_dataset("obstacle_distance", data=np.full(T, 100.0, dtype=np.float32))
        f.create_dataset("ego_x", data=np.zeros(T, dtype=np.float32))
        f.create_dataset("ego_y", data=np.zeros(T, dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.zeros(T, dtype=np.float32))

        if cf_flags:
            f.create_dataset("is_counterfactual", data=np.ones(T, dtype=bool))
            f.create_dataset("cf_is_branch", data=np.zeros(T, dtype=bool))
            f.create_dataset("cf_is_context", data=np.zeros(T, dtype=bool))
            f.create_dataset("cf_anchor_id", data=np.zeros(T, dtype=np.int64))
            f.create_dataset("cf_branch_step", data=np.zeros(T, dtype=np.int32))
            f.create_dataset("cf_steer_bias", data=np.zeros(T, dtype=np.float32))
            f.create_dataset("cf_branch_collision_diag", data=np.zeros(T, dtype=bool))
            f.create_dataset("cf_branch_lane_invasion_diag", data=np.zeros(T, dtype=bool))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema compatibility (existing + new dim checks)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaCheck:
    def test_passes_when_all_required_keys_present(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 10, 2)
        check_schema_compatibility(exp, cf)  # must not raise

    def test_raises_when_expert_missing_key(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
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
        with h5py.File(cf, "w") as f:
            f.create_dataset("images", data=np.zeros((4, 3, 64, 64), dtype=np.float32))
            f.create_dataset("states", data=np.zeros((4, 4), dtype=np.float32))
            f.create_dataset("actions", data=np.zeros((4, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="cf.*missing.*episode_ids"):
            check_schema_compatibility(exp, cf)

    def test_raises_on_state_dim_mismatch(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 4, 1, state_dim=4)
        _make_h5(cf, 4, 1, state_dim=6)
        with pytest.raises(ValueError, match="states.*last-dim mismatch"):
            check_schema_compatibility(exp, cf)

    def test_raises_on_action_dim_mismatch(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 4, 1, action_dim=2)
        _make_h5(cf, 4, 1, action_dim=3)
        with pytest.raises(ValueError, match="actions.*last-dim mismatch"):
            check_schema_compatibility(exp, cf)

    def test_raises_on_image_shape_mismatch(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        _make_h5(exp, 4, 1, image_hw=64)
        _make_h5(cf, 4, 1, image_hw=32)
        with pytest.raises(ValueError, match="images shape mismatch"):
            check_schema_compatibility(exp, cf)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Episode-level CF sampling
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeSampling:
    def _make_cf_data(self, n_eps: int, frames_per_ep: int) -> dict[str, np.ndarray]:
        T = n_eps * frames_per_ep
        ep_ids = np.repeat(np.arange(n_eps), frames_per_ep).astype(np.int64)
        return {
            "images": np.zeros((T, 3, 64, 64), dtype=np.float32),
            "episode_ids": ep_ids,
        }

    def test_selected_episodes_are_complete(self):
        """Each episode in the output must have exactly frames_per_ep frames."""
        n_eps, fpep = 10, 50
        cf_data = self._make_cf_data(n_eps, fpep)
        rng = np.random.default_rng(0)
        out = sample_cf_frames(cf_data, 0.5, rng)
        ep_ids = out["episode_ids"]
        for eid in np.unique(ep_ids):
            assert (ep_ids == eid).sum() == fpep, (
                f"Episode {eid} has {(ep_ids == eid).sum()} frames, expected {fpep}"
            )

    def test_no_partial_episodes(self):
        """No episode that was NOT selected should appear in the output."""
        n_eps, fpep = 8, 20
        cf_data = self._make_cf_data(n_eps, fpep)
        rng = np.random.default_rng(42)
        out = sample_cf_frames(cf_data, 0.5, rng)
        selected = set(np.unique(out["episode_ids"]).tolist())
        all_eps = set(range(n_eps))
        # Episode IDs in output must be a strict subset of all episodes
        assert selected.issubset(all_eps)
        # Frame count = exactly n_selected * fpep
        assert len(out["episode_ids"]) == len(selected) * fpep

    def test_frame_count_is_multiple_of_frames_per_ep(self):
        """n_cf_frames_sampled must be divisible by frames_per_ep (no partial sequences)."""
        n_eps, fpep = 12, 50
        cf_data = self._make_cf_data(n_eps, fpep)
        rng = np.random.default_rng(7)
        out = sample_cf_frames(cf_data, 0.3, rng)
        assert len(out["episode_ids"]) % fpep == 0

    def test_ratio_one_returns_all_episodes(self):
        n_eps, fpep = 5, 10
        cf_data = self._make_cf_data(n_eps, fpep)
        rng = np.random.default_rng(0)
        out = sample_cf_frames(cf_data, 1.0, rng)
        assert len(out["episode_ids"]) == n_eps * fpep

    def test_ratio_zero_returns_empty(self):
        n_eps, fpep = 5, 10
        cf_data = self._make_cf_data(n_eps, fpep)
        rng = np.random.default_rng(0)
        out = sample_cf_frames(cf_data, 0.0, rng)
        assert len(out["episode_ids"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. is_counterfactual flag correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestIsCounterfactualFlag:
    def test_expert_frames_all_false(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 20, 4)
        _make_h5(cf, 10, 2, cf_flags=True)
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            is_cf = f["is_counterfactual"][:].astype(bool)
            ep_ids = f["episode_ids"][:]
        # Expert episode_ids are 0..3; CF episodes are offset to 4..5
        expert_mask = ep_ids < 4
        assert not is_cf[expert_mask].any(), "Expert frames must have is_counterfactual=False"

    def test_cf_frames_all_true(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 20, 4)
        _make_h5(cf, 10, 2, cf_flags=True)
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            is_cf = f["is_counterfactual"][:].astype(bool)
            ep_ids = f["episode_ids"][:]
        cf_mask = ep_ids >= 4
        assert is_cf[cf_mask].all(), "CF frames must have is_counterfactual=True"

    def test_flag_synthesised_when_absent_from_cf_file(self, tmp_path):
        """CF file without is_counterfactual key → merge synthesises True."""
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 10, 2, cf_flags=False)  # no is_counterfactual in CF file
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            assert "is_counterfactual" in f
            is_cf = f["is_counterfactual"][:].astype(bool)
        # First 10 frames expert (False), last 10 CF (True)
        assert not is_cf[:10].any()
        assert is_cf[10:].all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. All keys padded → lengths consistent
# ─────────────────────────────────────────────────────────────────────────────

class TestAllKeysPadding:
    def test_all_keys_have_length_total_frames(self, tmp_path):
        """Every dataset in the merged file must have length == total_frames."""
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 20, 4)
        _make_h5(cf, 30, 6, cf_flags=True)
        summary = merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        total = summary["total_frames"]
        with h5py.File(out, "r") as f:
            for key in f.keys():
                assert f[key].shape[0] == total, (
                    f"Key '{key}': length {f[key].shape[0]} != total_frames {total}"
                )

    def test_cf_flag_keys_present_in_merged(self, tmp_path):
        """CF flag keys must exist in merged output even if expert has none."""
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 10, 2, cf_flags=True)
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            for k in CF_FLAG_KEYS:
                assert k in f, f"CF flag key '{k}' missing from merged output"

    def test_expert_cf_padding_is_zero(self, tmp_path):
        """Expert rows in cf_branch_step must be zero (padded, not garbage)."""
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        n_exp = 20
        _make_h5(exp, n_exp, 4)
        _make_h5(cf, 10, 2, cf_flags=True)
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            branch_step = f["cf_branch_step"][:n_exp]
        assert (branch_step == 0).all(), "Expert padding for cf_branch_step must be 0"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Episode-id offset (existing)
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodeIdOffset:
    def test_no_collision_after_merge(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)   # expert ids: 0, 1
        _make_h5(cf, 12, 3)    # cf ids: 0,1,2 → offset → 2,3,4
        merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0)
        with h5py.File(out, "r") as f:
            ep_ids = f["episode_ids"][:]
        expert_ids = set(range(2))
        cf_ids_after = set(range(2, 5))
        unique_ids = set(ep_ids.tolist())
        assert expert_ids.issubset(unique_ids)
        assert cf_ids_after.issubset(unique_ids)
        assert expert_ids.isdisjoint(cf_ids_after)

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


# ─────────────────────────────────────────────────────────────────────────────
# 6. cf_ratio sampling (episode-level, updated expectations)
# ─────────────────────────────────────────────────────────────────────────────

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

    def test_cf_ratio_half_episode_granularity(self, tmp_path):
        """0.5 ratio with 10 episodes × 10 frames = 50 CF frames."""
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        _make_h5(cf, 100, 10)  # 10 episodes × 10 frames
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. drop_collision_branches
# ─────────────────────────────────────────────────────────────────────────────

class TestDropCollisionBranches:
    def _make_cf_with_collision(self, path: Path, n_eps: int, fpep: int,
                                 collision_eps: list[int]) -> None:
        T = n_eps * fpep
        ep_ids = np.repeat(np.arange(n_eps), fpep).astype(np.int64)
        collision = np.zeros(T, dtype=bool)
        for eid in collision_eps:
            collision[ep_ids == eid] = True
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=np.zeros((T, 3, 64, 64), dtype=np.float32))
            f.create_dataset("states", data=np.zeros((T, 4), dtype=np.float32))
            f.create_dataset("actions", data=np.zeros((T, 2), dtype=np.float32))
            f.create_dataset("episode_ids", data=ep_ids)
            f.create_dataset("is_counterfactual", data=np.ones(T, dtype=bool))
            f.create_dataset("cf_branch_collision_diag", data=collision)

    def test_collision_episodes_excluded(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        n_eps, fpep = 6, 10
        collision_eps = [1, 3]  # 2 collision episodes
        self._make_cf_with_collision(cf, n_eps, fpep, collision_eps)
        summary = merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0,
                                 drop_collision_branches=True)
        # 6 - 2 = 4 episodes × 10 frames = 40 CF frames
        assert summary["n_cf_frames_sampled"] == 40
        assert summary["n_cf_frames_after_collision_filter"] == 40

    def test_drop_false_keeps_collision_episodes(self, tmp_path):
        exp = tmp_path / "exp.h5"
        cf = tmp_path / "cf.h5"
        out = tmp_path / "merged.h5"
        _make_h5(exp, 10, 2)
        n_eps, fpep = 4, 10
        self._make_cf_with_collision(cf, n_eps, fpep, [0, 2])
        summary = merge_datasets(exp, cf, out, cf_ratio=1.0, seed=0,
                                 drop_collision_branches=False)
        assert summary["n_cf_frames_sampled"] == n_eps * fpep  # all kept

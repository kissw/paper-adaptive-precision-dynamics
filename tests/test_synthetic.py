import numpy as np
import pytest
from active_inference.data.synthetic import SyntheticDrivingData


def test_synthetic_generation():
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=3, episode_len=50)

    total = 3 * 50
    assert data["images"].shape == (total, 3, 64, 64)
    assert data["states"].shape == (total, 2)
    assert data["actions"].shape == (total, 2)
    assert data["episode_ids"].shape == (total,)

    assert not np.isnan(data["images"]).any()
    assert not np.isnan(data["states"]).any()
    assert not np.isnan(data["actions"]).any()

    assert data["images"].dtype == np.float32
    assert data["states"].dtype == np.float32
    assert data["actions"].dtype == np.float32
    assert data["episode_ids"].dtype == np.int64


def test_hdf5_roundtrip(tmp_path):
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=2, episode_len=30)
    path = str(tmp_path / "test.h5")
    gen.to_hdf5(data, path)
    loaded = SyntheticDrivingData.from_hdf5(path)

    for key in data:
        np.testing.assert_array_equal(data[key], loaded[key])


def test_synthetic_action_range():
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=5, episode_len=100)
    actions = data["actions"]
    assert actions.min() >= -1.0
    assert actions.max() <= 1.0


def test_synthetic_episode_ids():
    gen = SyntheticDrivingData()
    n_episodes = 4
    episode_len = 25
    data = gen.generate(n_episodes=n_episodes, episode_len=episode_len)
    episode_ids = data["episode_ids"]

    for ep in range(n_episodes):
        start = ep * episode_len
        end = start + episode_len
        assert np.all(episode_ids[start:end] == ep)

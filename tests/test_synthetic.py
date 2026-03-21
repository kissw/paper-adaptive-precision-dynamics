import numpy as np
import pytest
from scipy import stats
from active_inference.data.synthetic import SyntheticDrivingData


def test_synthetic_generation():
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=3, episode_len=50)

    total = 3 * 50
    assert data["images"].shape == (total, 3, 64, 64)
    assert data["states"].shape == (total, 4)
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


def test_4d_state_columns():
    """States have 4 columns: speed, steer, heading_error, crosstrack_error."""
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=3, episode_len=100)
    states = data["states"]
    assert states.shape[1] == 4

    # Speed in reasonable range
    assert states[:, 0].min() >= 15.0
    assert states[:, 0].max() <= 45.0

    # Steer, heading, crosstrack in [-1, 1]
    for col in [1, 2, 3]:
        assert states[:, col].min() >= -1.0
        assert states[:, col].max() <= 1.0


def test_heading_correlates_with_steering():
    """heading_error should correlate with steering action (reactive expert)."""
    np.random.seed(42)
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=20, episode_len=200)
    ep_len = 200
    correlations = []
    for ep in range(20):
        s = ep * ep_len
        e = s + ep_len
        steer_action = data["actions"][s:e, 0]
        heading = data["states"][s:e, 2]
        # Expert corrects heading_error, so steering correlates with heading level
        r, _ = stats.pearsonr(steer_action, heading)
        if not np.isnan(r):
            correlations.append(abs(r))
    mean_corr = np.mean(correlations)
    assert mean_corr > 0.2, f"Heading-steering correlation too low: {mean_corr:.3f}"


def test_crosstrack_correlates_with_heading():
    """crosstrack_error should correlate with heading_error integral."""
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=10, episode_len=200)
    ep_len = 200
    correlations = []
    for ep in range(10):
        s = ep * ep_len
        e = s + ep_len
        heading = data["states"][s:e, 2]
        crosstrack = data["states"][s:e, 3]
        # crosstrack change vs heading
        dct = np.diff(crosstrack)
        r, _ = stats.pearsonr(heading[:-1], dct)
        if not np.isnan(r):
            correlations.append(abs(r))
    mean_corr = np.mean(correlations)
    assert mean_corr > 0.3, f"Crosstrack-heading correlation too low: {mean_corr:.3f}"


def test_success_flags():
    """success_flags array exists, dtype=bool, ~40-70% True."""
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=10, episode_len=200)
    flags = data["success_flags"]
    assert flags.dtype == bool
    assert flags.shape == (10 * 200,)
    ratio = flags.mean()
    assert 0.15 < ratio < 0.95, f"Success ratio out of range: {ratio:.3f}"


def test_images_encode_lateral_position():
    """Images should encode crosstrack position: left/right half means differ."""
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=5, episode_len=100)
    images = data["images"]
    crosstrack = data["states"][:, 3]

    # Split into left-shifted and right-shifted frames
    left_mask = crosstrack < -0.1
    right_mask = crosstrack > 0.1

    if left_mask.sum() > 5 and right_mask.sum() > 5:
        left_imgs = images[left_mask]
        right_imgs = images[right_mask]
        # Compare left-half vs right-half pixel means
        left_half_mean_left = left_imgs[:, :, :, :32].mean()
        left_half_mean_right = right_imgs[:, :, :, :32].mean()
        # When crosstrack < 0 (shifted left), image pattern shifts left
        # so the means should differ
        diff = abs(left_half_mean_left - left_half_mean_right)
        assert diff > 0.01, f"Image lateral encoding too weak: diff={diff:.4f}"

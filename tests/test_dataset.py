import numpy as np
import torch
import pytest
from active_inference.data.synthetic import SyntheticDrivingData
from active_inference.data.dataset import SequenceDataset, get_dataloader


@pytest.fixture
def small_h5(tmp_path):
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=3, episode_len=50)
    path = str(tmp_path / "small.h5")
    gen.to_hdf5(data, path)
    return path


def test_dataset_sequence_extraction(small_h5):
    seq_len = 20
    ds = SequenceDataset(small_h5, seq_len=seq_len)
    images, states, actions, *_ = ds[0]
    assert images.shape == (seq_len, 3, 64, 64)
    assert states.shape == (seq_len, 4)
    assert actions.shape == (seq_len, 2)
    assert isinstance(images, torch.Tensor)
    assert isinstance(states, torch.Tensor)
    assert isinstance(actions, torch.Tensor)


def test_dataset_episode_boundaries(tmp_path):
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=2, episode_len=20)
    path = str(tmp_path / "boundary.h5")
    gen.to_hdf5(data, path)

    seq_len = 15
    ds = SequenceDataset(path, seq_len=seq_len)

    for i in range(len(ds)):
        start = ds.valid_starts[i]
        end = start + seq_len - 1
        assert ds.episode_ids[start] == ds.episode_ids[end], (
            f"Sequence at {start} crosses episode boundary"
        )


def test_dataloader_batching(small_h5):
    seq_len = 10
    batch_size = 4
    dl = get_dataloader(
        small_h5, batch_size=batch_size, seq_len=seq_len, num_workers=0, shuffle=False
    )
    batch = next(iter(dl))
    images, states, actions, *_ = batch
    assert images.shape == (batch_size, seq_len, 3, 64, 64)
    assert states.shape == (batch_size, seq_len, 4)
    assert actions.shape == (batch_size, seq_len, 2)


def test_dataset_len(tmp_path):
    n_episodes = 2
    episode_len = 20
    seq_len = 15
    gen = SyntheticDrivingData()
    data = gen.generate(n_episodes=n_episodes, episode_len=episode_len)
    path = str(tmp_path / "len_test.h5")
    gen.to_hdf5(data, path)

    ds = SequenceDataset(path, seq_len=seq_len)
    expected = n_episodes * (episode_len - seq_len + 1)
    assert len(ds) == expected

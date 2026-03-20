import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, h5_path: str, seq_len: int = 50):
        self.h5_path = h5_path
        self.seq_len = seq_len

        with h5py.File(h5_path, "r") as f:
            self.total_samples = len(f["episode_ids"])
            self.episode_ids = f["episode_ids"][:]

        self.valid_starts = []
        for i in range(self.total_samples - seq_len + 1):
            if self.episode_ids[i] == self.episode_ids[i + seq_len - 1]:
                self.valid_starts.append(i)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len

        with h5py.File(self.h5_path, "r") as f:
            images = torch.from_numpy(f["images"][start:end])
            states = torch.from_numpy(f["states"][start:end])
            actions = torch.from_numpy(f["actions"][start:end])

        return images, states, actions


class PreferenceSequenceDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        seq_len: int = 50,
        task_filter: int | None = None,
        success_only: bool = True,
    ):
        self.h5_path = h5_path
        self.seq_len = seq_len

        with h5py.File(h5_path, "r") as f:
            self.total_samples = len(f["episode_ids"])
            episode_ids = f["episode_ids"][:]

            has_success = "success_flags" in f
            has_task = "task_labels" in f
            success_flags = (
                f["success_flags"][:] if has_success else np.ones(self.total_samples, dtype=bool)
            )
            task_labels = (
                f["task_labels"][:] if has_task else np.zeros(self.total_samples, dtype=np.int8)
            )

        frame_mask = np.ones(self.total_samples, dtype=bool)
        if success_only and has_success:
            frame_mask &= success_flags
        if task_filter is not None and has_task:
            frame_mask &= task_labels == task_filter

        self.valid_starts = []
        for i in range(self.total_samples - seq_len + 1):
            if episode_ids[i] != episode_ids[i + seq_len - 1]:
                continue
            if not frame_mask[i]:
                continue
            self.valid_starts.append(i)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len

        with h5py.File(self.h5_path, "r") as f:
            images = torch.from_numpy(f["images"][start:end])
            states = torch.from_numpy(f["states"][start:end])
            actions = torch.from_numpy(f["actions"][start:end])

        return images, states, actions


def get_dataloader(
    h5_path: str, batch_size=32, seq_len=50, num_workers=4, shuffle=True
) -> DataLoader:
    dataset = SequenceDataset(h5_path, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_preference_dataloader(
    h5_path: str,
    batch_size: int = 16,
    seq_len: int = 50,
    task_filter: int | None = None,
    success_only: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PreferenceSequenceDataset(
        h5_path,
        seq_len=seq_len,
        task_filter=task_filter,
        success_only=success_only,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

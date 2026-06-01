import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, h5_path: str, seq_len: int = 50):
        self.seq_len = seq_len

        with h5py.File(h5_path, "r") as f:
            self.total_samples = len(f["episode_ids"])
            self.episode_ids = f["episode_ids"][:]
            self._images = torch.from_numpy(f["images"][:])
            self._states = torch.from_numpy(f["states"][:])
            self._actions = torch.from_numpy(f["actions"][:])

            # Frame-level obstacle labels for auxiliary obstacle prediction.
            #
            # Priority:
            #   1. obstacle_visible: true frame-level visual label from v5.
            #   2. task_labels: legacy scenario/episode-level label.
            #   3. zeros: obstacle-free or unlabeled legacy data.
            #
            # Do not prefer task_labels when obstacle_visible exists. In v5,
            # task_labels == 1 means "obstacle scenario episode", while
            # obstacle_visible == 1 means "obstacle is visible in this frame".
            if "obstacle_visible" in f:
                self._obstacle_labels = torch.from_numpy(
                    f["obstacle_visible"][:].astype(np.float32)
                )
            elif "task_labels" in f:
                self._obstacle_labels = torch.from_numpy(
                    f["task_labels"][:].astype(np.float32)
                )
            else:
                self._obstacle_labels = torch.zeros(
                    self.total_samples,
                    dtype=torch.float32,
                )

        self.valid_starts = []
        for i in range(self.total_samples - seq_len + 1):
            if self.episode_ids[i] == self.episode_ids[i + seq_len - 1]:
                self.valid_starts.append(i)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len
        return (
            self._images[start:end],
            self._states[start:end],
            self._actions[start:end],
            self._obstacle_labels[start:end],
        )


class PreferenceSequenceDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        seq_len: int = 50,
        task_filter: int | None = None,
        success_only: bool = True,
    ):
        self.seq_len = seq_len

        with h5py.File(h5_path, "r") as f:
            self.total_samples = len(f["episode_ids"])
            episode_ids = f["episode_ids"][:]

            # Preload all data into memory.
            self._images = torch.from_numpy(f["images"][:])
            self._states = torch.from_numpy(f["states"][:])
            self._actions = torch.from_numpy(f["actions"][:])

            has_success = "success_flags" in f
            has_task = "task_labels" in f
            success_flags = (
                f["success_flags"][:]
                if has_success
                else np.ones(self.total_samples, dtype=bool)
            )
            task_labels = (
                f["task_labels"][:]
                if has_task
                else np.zeros(self.total_samples, dtype=np.int8)
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
        return self._images[start:end], self._states[start:end], self._actions[start:end]


def get_dataloader(
    h5_path: str,
    batch_size=32,
    seq_len=50,
    num_workers=4,
    shuffle=True,
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
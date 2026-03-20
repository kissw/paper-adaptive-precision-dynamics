import numpy as np
import h5py
import torch


class SyntheticDrivingData:
    def generate(self, n_episodes=5, episode_len=100) -> dict:
        all_images = []
        all_states = []
        all_actions = []
        all_episode_ids = []

        for ep in range(n_episodes):
            T = episode_len
            t = np.arange(T, dtype=np.float32)

            images = torch.randn(T, 3, 64, 64).clamp(-0.5, 0.5).numpy().astype(np.float32)

            speed = (30.0 + np.random.randn(T) * 2.0).astype(np.float32)
            freq = np.random.uniform(0.5, 2.0)
            steer = (np.sin(2 * np.pi * t / T * freq) + np.random.randn(T) * 0.05).astype(
                np.float32
            )
            states = np.stack([speed, steer], axis=1)

            steer_cmd = (steer + np.random.randn(T) * 0.02).astype(np.float32)
            accel_base = np.random.uniform(0.3, 0.7, size=T).astype(np.float32)
            brake_mask = np.random.rand(T) < 0.1
            accel_cmd = accel_base.copy()
            accel_cmd[brake_mask] = np.random.uniform(-0.5, -0.1, size=brake_mask.sum()).astype(
                np.float32
            )
            accel_cmd = np.clip(accel_cmd, -1.0, 1.0).astype(np.float32)
            steer_cmd = np.clip(steer_cmd, -1.0, 1.0).astype(np.float32)
            actions = np.stack([steer_cmd, accel_cmd], axis=1)

            episode_ids = np.full(T, ep, dtype=np.int64)

            all_images.append(images)
            all_states.append(states)
            all_actions.append(actions)
            all_episode_ids.append(episode_ids)

        return {
            "images": np.concatenate(all_images, axis=0),
            "states": np.concatenate(all_states, axis=0),
            "actions": np.concatenate(all_actions, axis=0),
            "episode_ids": np.concatenate(all_episode_ids, axis=0),
        }

    def to_hdf5(self, data: dict, path: str) -> None:
        with h5py.File(path, "w") as f:
            for key, arr in data.items():
                f.create_dataset(key, data=arr)

    @staticmethod
    def from_hdf5(path: str) -> dict:
        result = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                result[key] = f[key][:]
        return result

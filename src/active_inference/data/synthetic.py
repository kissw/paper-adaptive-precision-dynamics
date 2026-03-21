import numpy as np
import h5py
import torch


class SyntheticDrivingData:
    def generate(self, n_episodes=5, episode_len=100) -> dict:
        all_images = []
        all_states = []
        all_actions = []
        all_episode_ids = []
        all_success_flags = []

        for ep in range(n_episodes):
            T = episode_len
            t = np.arange(T, dtype=np.float32)
            dt = 0.05

            # Road curvature varies sinusoidally
            curve_freq = np.random.uniform(0.5, 2.0)
            curve_amp = np.random.uniform(0.3, 0.8)
            road_curvature = (curve_amp * np.sin(2 * np.pi * t / T * curve_freq)).astype(
                np.float32
            )

            # Wind/disturbance that pushes heading (simulates road camber, gusts)
            dist_freq = np.random.uniform(1.0, 4.0)
            dist_amp = np.random.uniform(0.3, 0.6)
            disturbance = (dist_amp * np.sin(2 * np.pi * t / T * dist_freq)).astype(
                np.float32
            )

            # Speed with gentle variation
            speed = (30.0 + np.random.randn(T).cumsum() * 0.3).clip(20, 40).astype(np.float32)

            # Coupled kinematic simulation
            heading_error = np.zeros(T, dtype=np.float32)
            crosstrack_error = np.zeros(T, dtype=np.float32)
            steer_cmd = np.zeros(T, dtype=np.float32)

            # Initial perturbation
            heading_error[0] = np.random.randn() * 0.3
            crosstrack_error[0] = np.random.randn() * 0.2

            # Expert correction gains
            k_heading = 1.5
            k_crosstrack = 0.8
            noise_scale = 0.1

            # Dynamics gains
            heading_gain = 6.0   # how much steering affects heading
            crosstrack_gain = 8.0  # how much heading affects crosstrack

            for i in range(T):
                # Expert steering: follow curvature + correct errors
                steer_cmd[i] = (
                    road_curvature[i]
                    - k_heading * heading_error[i]
                    - k_crosstrack * crosstrack_error[i]
                    + np.random.randn() * noise_scale
                )
                steer_cmd[i] = np.clip(steer_cmd[i], -1.0, 1.0)

                if i < T - 1:
                    # Heading changes from steering mismatch + disturbance
                    heading_error[i + 1] = heading_error[i] + (
                        steer_cmd[i] - road_curvature[i]
                    ) * dt * heading_gain + disturbance[i] * dt * 3.0
                    heading_error[i + 1] += np.random.randn() * 0.02
                    heading_error[i + 1] = np.clip(heading_error[i + 1], -1.0, 1.0)

                    # Crosstrack accumulates from heading offset
                    crosstrack_error[i + 1] = (
                        crosstrack_error[i] + heading_error[i] * dt * crosstrack_gain
                    )
                    crosstrack_error[i + 1] += np.random.randn() * 0.01
                    crosstrack_error[i + 1] = np.clip(crosstrack_error[i + 1], -1.0, 1.0)

            # Observed steering = commanded + small lag
            steer_obs = (steer_cmd * 0.9 + np.roll(steer_cmd, 1) * 0.1).astype(np.float32)
            steer_obs[0] = steer_cmd[0]

            # 4D state: [speed, steer, heading_error, crosstrack_error]
            states = np.stack([speed, steer_obs, heading_error, crosstrack_error], axis=1)

            # Success: frames near lane center
            success = np.abs(crosstrack_error) < 0.15

            # Actions: [steer_cmd, accel_cmd]
            accel_base = np.random.uniform(0.3, 0.7, size=T).astype(np.float32)
            brake_mask = np.random.rand(T) < 0.1
            accel_cmd = accel_base.copy()
            accel_cmd[brake_mask] = np.random.uniform(
                -0.5, -0.1, size=brake_mask.sum()
            ).astype(np.float32)
            accel_cmd = np.clip(accel_cmd, -1.0, 1.0).astype(np.float32)
            actions = np.stack([steer_cmd, accel_cmd], axis=1)

            # Lane-like images: horizontal gradient encodes crosstrack position
            images = np.zeros((T, 3, 64, 64), dtype=np.float32)
            x_coords = np.linspace(-1, 1, 64).astype(np.float32)
            road_start = int(64 * 0.4)
            for i in range(T):
                shift = crosstrack_error[i]
                for c in range(3):
                    images[i, c, :road_start, :] = np.random.randn(road_start, 64) * 0.05 - 0.3
                    for row in range(road_start, 64):
                        frac = (row - road_start) / (64 - road_start)
                        squeeze = 0.3 + 0.7 * frac
                        squeezed_x = x_coords / squeeze
                        g = np.exp(-((squeezed_x - shift) ** 2) / 0.5)
                        ll = np.exp(-((squeezed_x - shift - 0.4) ** 2) / 0.02)
                        lr = np.exp(-((squeezed_x - shift + 0.4) ** 2) / 0.02)
                        rp = g + 0.5 * (ll + lr)
                        rp = rp / (rp.max() + 1e-8) - 0.5
                        images[i, c, row, :] = rp + np.random.randn(64) * 0.02
                images[i, 0] *= 0.9
                images[i, 2] *= 1.1

            images = np.clip(images, -0.5, 0.5).astype(np.float32)
            episode_ids = np.full(T, ep, dtype=np.int64)

            all_images.append(images)
            all_states.append(states)
            all_actions.append(actions)
            all_episode_ids.append(episode_ids)
            all_success_flags.append(success)

        return {
            "images": np.concatenate(all_images, axis=0),
            "states": np.concatenate(all_states, axis=0),
            "actions": np.concatenate(all_actions, axis=0),
            "episode_ids": np.concatenate(all_episode_ids, axis=0),
            "success_flags": np.concatenate(all_success_flags, axis=0),
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

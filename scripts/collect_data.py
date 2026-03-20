import argparse
import sys
from pathlib import Path

import numpy as np
import h5py


# --- Ornstein-Uhlenbeck noise for temporally correlated action perturbation ---
# dx = theta * (mu - x) * dt + sigma * sqrt(dt) * N(0,1)
# Produces sustained deviations that create realistic maneuvers + recovery arcs
class OUNoise:
    def __init__(
        self, dim: int = 2, sigma: list[float] | None = None, theta: float = 0.15, dt: float = 0.05
    ):
        self._theta = theta
        self._sigma = np.array(sigma if sigma is not None else [0.15, 0.10])
        self._dt = dt
        self._x = np.zeros(dim)

    def sample(self) -> np.ndarray:
        dx = self._theta * (0.0 - self._x) * self._dt + self._sigma * np.sqrt(
            self._dt
        ) * np.random.randn(len(self._x))
        self._x += dx
        return self._x.copy()

    def reset(self):
        self._x = np.zeros_like(self._x)


# --- 3-tier noise allocation ---
# Tier distribution: 25% clean, 40% medium, 30% high, 5% random burst
TIERS = {
    "clean": {"sigma": [0.0, 0.0], "weight": 0.25},
    "medium": {"sigma": [0.15, 0.10], "weight": 0.40},
    "high": {"sigma": [0.30, 0.20], "weight": 0.30},
    "random": {"sigma": [1.0, 1.0], "weight": 0.05},
}


def select_tier(rng: np.random.Generator) -> str:
    names = list(TIERS.keys())
    weights = [TIERS[n]["weight"] for n in names]
    return rng.choice(names, p=weights)


def main():
    parser = argparse.ArgumentParser(description="Collect diverse expert driving data from CARLA")
    parser.add_argument("--town", default="Town06")
    parser.add_argument("--num_samples", type=int, default=72000)
    parser.add_argument("--output", default="data/expert_data.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=1000)
    parser.add_argument("--num_npcs", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_speed", type=float, default=15.0)
    parser.add_argument("--max_speed", type=float, default=50.0)
    args = parser.parse_args()

    try:
        import carla
        from agents.navigation.basic_agent import BasicAgent
    except ImportError:
        print("ERROR: carla package not installed. Install with: pip install carla==0.9.16")
        sys.exit(1)

    from active_inference.data.carla_env import CARLADrivingEnv, carla_to_action

    rng = np.random.default_rng(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = CARLADrivingEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        image_model_size=args.image_size,
    )

    all_images = []
    all_states = []
    all_actions = []
    all_expert_actions = []
    all_episode_ids = []
    all_lateral_devs = []
    all_lane_ids = []
    all_noise_sigmas = []

    # Per-episode metadata (backfilled after episode)
    ep_success_flags = []
    ep_task_labels = []
    ep_target_speeds = []
    ep_tiers = []

    episode_idx = 0
    collected = 0
    tier_counts = {k: 0 for k in TIERS}

    try:
        while collected < args.num_samples:
            obs = env.reset()
            img, state = obs

            tier = select_tier(rng)
            tier_counts[tier] += 1
            sigma = TIERS[tier]["sigma"]
            noise = OUNoise(dim=2, sigma=sigma)

            target_speed = float(rng.uniform(args.min_speed, args.max_speed))
            agent = BasicAgent(env._vehicle, target_speed=target_speed)
            spawn_points = env._world.get_map().get_spawn_points()
            dest = spawn_points[rng.integers(len(spawn_points))]
            agent.set_destination(dest.location)
            env.set_goal(dest.location)

            ep_start = collected
            ep_collisions = 0
            ep_lat_devs = []
            ep_lane_list = []

            for t in range(args.episode_len):
                if collected >= args.num_samples:
                    break

                control = agent.run_step()
                expert_action = carla_to_action(control.steer, control.throttle, control.brake)

                if tier == "random" and rng.random() < 0.3:
                    noisy_action = np.array([rng.uniform(-1, 1), rng.uniform(-0.3, 0.5)])
                else:
                    noisy_action = np.clip(expert_action + noise.sample(), -1.0, 1.0)
                    speed_kmh = state[0] * 3.6
                    if speed_kmh > target_speed * 1.1 and expert_action[1] < 0:
                        noisy_action[1] = min(noisy_action[1], expert_action[1])

                obs, info = env.step(noisy_action)
                img, state = obs

                all_images.append(img)
                all_states.append(state)
                all_actions.append(noisy_action)
                all_expert_actions.append(expert_action)
                all_episode_ids.append(episode_idx)
                all_noise_sigmas.append(sigma)

                if info.get("collision", False):
                    ep_collisions += 1

                waypoint = env._world.get_map().get_waypoint(env._vehicle.get_location())
                lane_id = waypoint.lane_id if waypoint else -1
                loc = env._vehicle.get_location()
                if waypoint:
                    wp_loc = waypoint.transform.location
                    lat_dev = ((loc.x - wp_loc.x) ** 2 + (loc.y - wp_loc.y) ** 2) ** 0.5
                else:
                    lat_dev = 0.0

                all_lateral_devs.append(lat_dev)
                all_lane_ids.append(lane_id)
                ep_lat_devs.append(lat_dev)
                ep_lane_list.append(lane_id)

                collected += 1

                if agent.done() or ep_collisions > 0:
                    break

            ep_end = collected
            ep_frames = ep_end - ep_start

            mean_lat = float(np.mean(ep_lat_devs)) if ep_lat_devs else 0.0
            is_success = ep_collisions == 0 and mean_lat < 0.5

            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_list))
                if ep_lane_list[i] != ep_lane_list[i - 1]
                and ep_lane_list[i] != -1
                and ep_lane_list[i - 1] != -1
            )
            task_label = 1 if lane_changes > 0 else 0

            for _ in range(ep_frames):
                ep_success_flags.append(is_success)
                ep_task_labels.append(task_label)
                ep_target_speeds.append(target_speed)
                ep_tiers.append(tier)

            episode_idx += 1
            task_str = "B(lane_change)" if task_label == 1 else "A(lane_keep)"
            print(
                f"Ep {episode_idx:4d} [{tier:6s}] spd={target_speed:4.0f} | "
                f"{ep_frames:4d}fr col={ep_collisions} lat={mean_lat:.3f} "
                f"task={task_str} ok={is_success} | {collected}/{args.num_samples}"
            )

    finally:
        env.close()

    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=np.stack(all_images), dtype=np.float32)
        f.create_dataset("states", data=np.stack(all_states), dtype=np.float32)
        f.create_dataset("actions", data=np.stack(all_actions), dtype=np.float32)
        f.create_dataset("expert_actions", data=np.stack(all_expert_actions), dtype=np.float32)
        f.create_dataset("episode_ids", data=np.array(all_episode_ids, dtype=np.int64))
        f.create_dataset("lateral_devs", data=np.array(all_lateral_devs, dtype=np.float32))
        f.create_dataset("lane_ids", data=np.array(all_lane_ids, dtype=np.int32))
        f.create_dataset("noise_sigmas", data=np.array(all_noise_sigmas, dtype=np.float32))
        f.create_dataset("success_flags", data=np.array(ep_success_flags, dtype=bool))
        f.create_dataset("task_labels", data=np.array(ep_task_labels, dtype=np.int8))
        f.create_dataset("target_speeds", data=np.array(ep_target_speeds, dtype=np.float32))

        tier_str = ", ".join(f"{k}={v}" for k, v in tier_counts.items())
        f.attrs["town"] = args.town
        f.attrs["total_frames"] = collected
        f.attrs["total_episodes"] = episode_idx
        f.attrs["tier_distribution"] = tier_str

    print(f"\nSaved {collected} frames ({episode_idx} episodes) to {args.output}")
    print(f"Tier distribution: {tier_str}")
    task_a = sum(1 for t in ep_task_labels if t == 0)
    task_b = sum(1 for t in ep_task_labels if t == 1)
    successes = sum(1 for s in ep_success_flags if s)
    print(f"Task A frames: {task_a}, Task B frames: {task_b}")
    print(f"Success frames: {successes}/{collected} ({100 * successes / max(collected, 1):.1f}%)")


if __name__ == "__main__":
    main()

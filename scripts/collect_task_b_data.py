"""Collect lane-change training data for Task B (obstacle avoidance).

Spawns obstacles on multi-lane highways and records BehaviorAgent
performing lane changes to avoid them. Filters for successful episodes
(collision-free with at least 1 lane change detected).

Usage:
    uv run python scripts/collect_task_b_data.py \
        --town Town06 --num_samples 15000 \
        --output data/task_b_lanechange.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py


# Noise tiers for Task B: clean(30%), medium(40%), high(30%)
TIERS = {
    "clean": {"sigma": [0.0, 0.0], "weight": 0.30},
    "medium": {"sigma": [0.15, 0.05], "weight": 0.40},
    "high": {"sigma": [0.30, 0.10], "weight": 0.30},
}


class OUNoise:
    def __init__(self, dim: int = 2, sigma: list[float] | None = None,
                 theta: float = 0.15, dt: float = 0.05):
        self._theta = theta
        self._sigma = np.array(sigma if sigma is not None else [0.15, 0.10])
        self._dt = dt
        self._x = np.zeros(dim)

    def sample(self) -> np.ndarray:
        dx = (self._theta * (0.0 - self._x) * self._dt
              + self._sigma * np.sqrt(self._dt) * np.random.randn(len(self._x)))
        self._x += dx
        return self._x.copy()

    def reset(self):
        self._x = np.zeros_like(self._x)


def select_tier(rng: np.random.Generator) -> str:
    names = list(TIERS.keys())
    weights = [TIERS[n]["weight"] for n in names]
    return rng.choice(names, p=weights)


def main():
    parser = argparse.ArgumentParser(
        description="Collect lane-change data for Task B obstacle avoidance"
    )
    parser.add_argument("--town", default="Town06")
    parser.add_argument("--num_samples", type=int, default=15000)
    parser.add_argument("--output", default="data/task_b_lanechange.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=600)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_speed", type=float, default=20.0,
                        help="Min target speed km/h")
    parser.add_argument("--max_speed", type=float, default=40.0,
                        help="Max target speed km/h")
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--obstacle_distances", type=int, nargs="+",
                        default=[40, 90, 140],
                        help="Waypoint-step distances for obstacles")
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("ERROR: carla package required. Install with: pip install carla==0.9.16")
        sys.exit(1)

    # Try BehaviorAgent first (better lane-change behavior), fall back to BasicAgent
    try:
        from agents.navigation.behavior_agent import BehaviorAgent
        USE_BEHAVIOR_AGENT = True
        print("Using BehaviorAgent (aggressive profile) for lane-change data")
    except ImportError:
        from agents.navigation.basic_agent import BasicAgent
        USE_BEHAVIOR_AGENT = False
        print("BehaviorAgent not available, falling back to BasicAgent")

    from active_inference.data.carla_env import CARLADrivingEnv, carla_to_action
    from active_inference.evaluation.obstacles import (
        spawn_obstacles_multilane,
        destroy_obstacles,
    )

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

    ep_success_flags = []
    ep_task_labels = []
    ep_target_speeds = []

    episode_idx = 0
    collected = 0
    tier_counts = {k: 0 for k in TIERS}
    stats = {
        "total_episodes": 0,
        "lane_change_episodes": 0,
        "collision_free_episodes": 0,
        "successful_episodes": 0,  # collision-free AND lane change
    }

    obstacle_actors = []

    try:
        while collected < args.num_samples:
            # Clean up previous obstacles
            destroy_obstacles(obstacle_actors)

            obs = env.reset()
            img, state = obs

            tier = select_tier(rng)
            tier_counts[tier] += 1
            sigma = TIERS[tier]["sigma"]
            noise = OUNoise(dim=2, sigma=sigma)

            # Target speed in m/s (args are in km/h)
            target_speed_kmh = float(rng.uniform(args.min_speed, args.max_speed))
            target_speed_ms = target_speed_kmh / 3.6

            # Create navigation agent
            if USE_BEHAVIOR_AGENT:
                agent = BehaviorAgent(env._vehicle, behavior="aggressive")
            else:
                agent = BasicAgent(env._vehicle, target_speed=target_speed_ms)

            # Set random destination
            spawn_points = env._world.get_map().get_spawn_points()
            dest = spawn_points[rng.integers(len(spawn_points))]
            agent.set_destination(dest.location)
            env.set_goal(dest.location)

            # Spawn obstacles on multi-lane positions
            obstacle_actors = spawn_obstacles_multilane(
                env._world,
                env._vehicle,
                num_obstacles=args.num_obstacles,
                distances=args.obstacle_distances,
            )

            ep_start = collected
            ep_collisions = 0
            ep_lat_devs = []
            ep_lane_list = []

            for t in range(args.episode_len):
                if collected >= args.num_samples:
                    break

                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake
                )

                # Apply noise perturbation
                noisy_action = np.clip(expert_action + noise.sample(), -1.0, 1.0)

                # Speed limiting for safer obstacle approach
                speed_kmh = state[0] * 3.6
                if speed_kmh > target_speed_kmh * 1.1 and expert_action[1] < 0:
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

                waypoint = env._world.get_map().get_waypoint(
                    env._vehicle.get_location()
                )
                lane_id = waypoint.lane_id if waypoint else -1
                loc = env._vehicle.get_location()
                if waypoint:
                    wp_loc = waypoint.transform.location
                    lat_dev = (
                        (loc.x - wp_loc.x) ** 2 + (loc.y - wp_loc.y) ** 2
                    ) ** 0.5
                else:
                    lat_dev = 0.0

                all_lateral_devs.append(lat_dev)
                all_lane_ids.append(lane_id)
                ep_lat_devs.append(lat_dev)
                ep_lane_list.append(lane_id)

                collected += 1

                # End episode on collision (but keep the data)
                if ep_collisions > 0:
                    break

                if agent.done():
                    break

            # Destroy obstacles at end of episode
            destroy_obstacles(obstacle_actors)

            ep_end = collected
            ep_frames = ep_end - ep_start

            # Detect lane changes
            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_list))
                if ep_lane_list[i] != ep_lane_list[i - 1]
                and ep_lane_list[i] != -1
                and ep_lane_list[i - 1] != -1
            )

            has_lane_change = lane_changes > 0
            is_collision_free = ep_collisions == 0
            is_success = is_collision_free and has_lane_change
            task_label = 1 if has_lane_change else 0

            # Update stats
            stats["total_episodes"] += 1
            if has_lane_change:
                stats["lane_change_episodes"] += 1
            if is_collision_free:
                stats["collision_free_episodes"] += 1
            if is_success:
                stats["successful_episodes"] += 1

            # Backfill per-frame metadata
            for _ in range(ep_frames):
                ep_success_flags.append(is_success)
                ep_task_labels.append(task_label)
                ep_target_speeds.append(target_speed_kmh)

            episode_idx += 1
            status = "OK" if is_success else ("COL" if not is_collision_free else "NO-LC")
            print(
                f"Ep {episode_idx:4d} [{tier:6s}] spd={target_speed_kmh:4.0f}km/h | "
                f"{ep_frames:4d}fr col={ep_collisions} lc={lane_changes} "
                f"status={status} obs={len(obstacle_actors)} | "
                f"{collected}/{args.num_samples}"
            )

    finally:
        destroy_obstacles(obstacle_actors)
        env.close()

    # Save to HDF5
    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=np.stack(all_images), dtype=np.float32)
        f.create_dataset("states", data=np.stack(all_states), dtype=np.float32)
        f.create_dataset("actions", data=np.stack(all_actions), dtype=np.float32)
        f.create_dataset(
            "expert_actions", data=np.stack(all_expert_actions), dtype=np.float32
        )
        f.create_dataset(
            "episode_ids", data=np.array(all_episode_ids, dtype=np.int64)
        )
        f.create_dataset(
            "lateral_devs", data=np.array(all_lateral_devs, dtype=np.float32)
        )
        f.create_dataset("lane_ids", data=np.array(all_lane_ids, dtype=np.int32))
        f.create_dataset(
            "noise_sigmas", data=np.array(all_noise_sigmas, dtype=np.float32)
        )
        f.create_dataset(
            "success_flags", data=np.array(ep_success_flags, dtype=bool)
        )
        f.create_dataset(
            "task_labels", data=np.array(ep_task_labels, dtype=np.int8)
        )
        f.create_dataset(
            "target_speeds", data=np.array(ep_target_speeds, dtype=np.float32)
        )

        tier_str = ", ".join(f"{k}={v}" for k, v in tier_counts.items())
        f.attrs["town"] = args.town
        f.attrs["total_frames"] = collected
        f.attrs["total_episodes"] = episode_idx
        f.attrs["tier_distribution"] = tier_str
        f.attrs["task"] = "B_lanechange"

    print(f"\nSaved {collected} frames ({episode_idx} episodes) to {args.output}")
    print(f"Tier distribution: {tier_str}")
    print(f"\nTask B Collection Stats:")
    print(f"  Total episodes:          {stats['total_episodes']}")
    print(f"  Lane-change episodes:    {stats['lane_change_episodes']} "
          f"({100 * stats['lane_change_episodes'] / max(stats['total_episodes'], 1):.1f}%)")
    print(f"  Collision-free episodes: {stats['collision_free_episodes']} "
          f"({100 * stats['collision_free_episodes'] / max(stats['total_episodes'], 1):.1f}%)")
    print(f"  Successful episodes:     {stats['successful_episodes']} "
          f"({100 * stats['successful_episodes'] / max(stats['total_episodes'], 1):.1f}%)")

    task_b_frames = sum(1 for t in ep_task_labels if t == 1)
    print(f"\n  Task B frames: {task_b_frames}/{collected} "
          f"({100 * task_b_frames / max(collected, 1):.1f}%)")


if __name__ == "__main__":
    main()

"""Collect clean normal-driving data (v5) on Town06_Opt same route as obstacle v5.

No obstacles are spawned.  BasicAgent drives Town06_Opt_TaskB routes and we
record all frames from completed (collision-free, goal-reached) episodes.

HDF5 schema is identical to collect_obstacle_data_v5.py; obstacle fields are
set to placeholder values (visible=False, distance=inf, bbox=NaN, etc.) so
that downstream processing can treat both files uniformly.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import h5py

# v5 obstacle metadata constants (shared schema)
_V5_KEYS_OBSTACLE = [
    "obstacle_visible", "obstacle_distance", "obstacle_in_front",
    "obstacle_lateral", "nearest_obstacle_id",
    "obstacle_bbox", "obstacle_bbox_area",
    "ego_x", "ego_y", "ego_yaw",
    "obstacle_x", "obstacle_y",
]

_V5_KEYS_BASE = [
    "images", "states", "actions", "expert_actions", "episode_ids",
    "lateral_devs", "lane_ids", "noise_sigmas", "success_flags", "task_labels",
]


def _clean_obstacle_frame_meta(ego_x: float, ego_y: float, ego_yaw: float) -> dict:
    """Obstacle metadata for a clean (no-obstacle) frame."""
    return {
        "obstacle_visible":    False,
        "obstacle_distance":   float("inf"),
        "obstacle_in_front":   False,
        "obstacle_lateral":    float("nan"),
        "nearest_obstacle_id": -1,
        "obstacle_bbox":       np.full(4, np.nan, dtype=np.float32),
        "obstacle_bbox_area":  0.0,
        "ego_x":               ego_x,
        "ego_y":               ego_y,
        "ego_yaw":             ego_yaw,
        "obstacle_x":          float("nan"),
        "obstacle_y":          float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect clean normal-driving data (v5) — no obstacles",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--output", default="data/expert_clean_v5.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_speed", type=float, default=25.0)
    args = parser.parse_args()

    try:
        import carla
        from agents.navigation.basic_agent import BasicAgent
    except ImportError:
        print("ERROR: carla package required")
        sys.exit(1)

    from active_inference.data.carla_env import CARLADrivingEnv, carla_to_action
    from active_inference.evaluation.routes import EVAL_ROUTES, get_route_waypoints

    rng = np.random.default_rng(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = CARLADrivingEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        image_model_size=args.image_size,
    )

    route_key = "Town06_Opt_TaskB"
    routes = EVAL_ROUTES.get(route_key, [])
    if not routes:
        print(f"ERROR: No routes for {route_key}")
        sys.exit(1)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    carla_map = client.get_world().get_map()

    # Accumulators
    all_images, all_states, all_actions, all_expert_actions = [], [], [], []
    all_episode_ids, all_lateral_devs, all_lane_ids = [], [], []
    all_noise_sigmas, ep_success_flags, ep_task_labels = [], [], []

    all_obs_visible, all_obs_distance, all_obs_in_front = [], [], []
    all_obs_lateral, all_nearest_obs_id, all_obs_bbox = [], [], []
    all_obs_bbox_area, all_ego_x, all_ego_y, all_ego_yaw = [], [], [], []
    all_obs_x, all_obs_y = [], []

    episode_idx = 0
    collected = 0
    good_episodes = 0
    total_attempts = 0

    try:
        while collected < args.num_samples:
            route_idx = rng.integers(len(routes))
            route = routes[route_idx]
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location

            obs = env.reset(spawn_point=start_spawn)
            img, state = obs
            env.set_goal(goal_loc)

            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            ep_images, ep_states, ep_actions, ep_expert_actions = [], [], [], []
            ep_lat_devs, ep_lane_list = [], []
            ep_collisions = 0
            ep_meta: list[dict] = []
            total_attempts += 1

            for _t in range(args.episode_len):
                loc = env._vehicle.get_location()
                ego_yaw = float(env._vehicle.get_transform().rotation.yaw)

                meta = _clean_obstacle_frame_meta(
                    float(loc.x), float(loc.y), ego_yaw,
                )
                ep_meta.append(meta)

                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake,
                )
                noise = rng.normal(0, 0.02, size=2)
                noisy_action = np.clip(expert_action + noise, -1.0, 1.0)

                obs, info = env.step(noisy_action)
                img, state = obs

                ep_images.append(img)
                ep_states.append(state)
                ep_actions.append(noisy_action)
                ep_expert_actions.append(expert_action)

                if info.get("collision", False):
                    ep_collisions += 1

                waypoint = carla_map.get_waypoint(loc)
                lane_id = waypoint.lane_id if waypoint else -1
                if waypoint:
                    wp_loc = waypoint.transform.location
                    lat_dev = math.sqrt(
                        (loc.x - wp_loc.x) ** 2 + (loc.y - wp_loc.y) ** 2
                    )
                else:
                    lat_dev = 0.0
                ep_lat_devs.append(lat_dev)
                ep_lane_list.append(lane_id)

                goal_dist = math.sqrt(
                    (loc.x - goal_loc.x) ** 2 + (loc.y - goal_loc.y) ** 2
                )
                if goal_dist < 15.0 or agent.done():
                    break
                if ep_collisions > 0:
                    break

            ep_frames = len(ep_images)
            is_good = ep_collisions == 0 and ep_frames > 50

            if is_good:
                for i in range(ep_frames):
                    m = ep_meta[i]
                    all_images.append(ep_images[i])
                    all_states.append(ep_states[i])
                    all_actions.append(ep_actions[i])
                    all_expert_actions.append(ep_expert_actions[i])
                    all_episode_ids.append(episode_idx)
                    all_lateral_devs.append(ep_lat_devs[i])
                    all_lane_ids.append(ep_lane_list[i])
                    all_noise_sigmas.append([0.02, 0.02])
                    ep_success_flags.append(True)
                    ep_task_labels.append(0)
                    all_obs_visible.append(m["obstacle_visible"])
                    all_obs_distance.append(m["obstacle_distance"])
                    all_obs_in_front.append(m["obstacle_in_front"])
                    all_obs_lateral.append(m["obstacle_lateral"])
                    all_nearest_obs_id.append(m["nearest_obstacle_id"])
                    all_obs_bbox.append(m["obstacle_bbox"])
                    all_obs_bbox_area.append(m["obstacle_bbox_area"])
                    all_ego_x.append(m["ego_x"])
                    all_ego_y.append(m["ego_y"])
                    all_ego_yaw.append(m["ego_yaw"])
                    all_obs_x.append(m["obstacle_x"])
                    all_obs_y.append(m["obstacle_y"])

                collected += ep_frames
                episode_idx += 1
                good_episodes += 1
                print(
                    f"  Ep {episode_idx:4d} [CLEAN] | {ep_frames:4d}fr "
                    f"| {collected}/{args.num_samples}"
                )
            else:
                reason = f"col={ep_collisions}" if ep_collisions > 0 else f"short={ep_frames}"
                print(f"  Attempt {total_attempts} [SKIP {reason}]")

            if collected >= args.num_samples:
                break

    finally:
        env.close()

    if not all_images:
        print("ERROR: No frames collected")
        sys.exit(1)

    n = min(len(all_images), args.num_samples)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=np.stack(all_images[:n]), dtype=np.float32)
        f.create_dataset("states", data=np.stack(all_states[:n]), dtype=np.float32)
        f.create_dataset("actions", data=np.stack(all_actions[:n]), dtype=np.float32)
        f.create_dataset("expert_actions", data=np.stack(all_expert_actions[:n]), dtype=np.float32)
        f.create_dataset("episode_ids", data=np.array(all_episode_ids[:n], dtype=np.int64))
        f.create_dataset("lateral_devs", data=np.array(all_lateral_devs[:n], dtype=np.float32))
        f.create_dataset("lane_ids", data=np.array(all_lane_ids[:n], dtype=np.int32))
        f.create_dataset("noise_sigmas", data=np.array(all_noise_sigmas[:n], dtype=np.float32))
        f.create_dataset("success_flags", data=np.array(ep_success_flags[:n], dtype=bool))
        f.create_dataset("task_labels", data=np.array(ep_task_labels[:n], dtype=np.int8))

        f.create_dataset("obstacle_visible", data=np.zeros(n, dtype=bool))
        dist_arr = np.full(n, np.inf, dtype=np.float32)
        f.create_dataset("obstacle_distance", data=dist_arr)
        f.create_dataset("obstacle_in_front", data=np.zeros(n, dtype=bool))
        f.create_dataset("obstacle_lateral", data=np.full(n, np.nan, dtype=np.float32))
        f.create_dataset("nearest_obstacle_id", data=np.full(n, -1, dtype=np.int32))
        f.create_dataset("obstacle_bbox", data=np.full((n, 4), np.nan, dtype=np.float32))
        f.create_dataset("obstacle_bbox_area", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("ego_x", data=np.array(all_ego_x[:n], dtype=np.float32))
        f.create_dataset("ego_y", data=np.array(all_ego_y[:n], dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.array(all_ego_yaw[:n], dtype=np.float32))
        f.create_dataset("obstacle_x", data=np.full(n, np.nan, dtype=np.float32))
        f.create_dataset("obstacle_y", data=np.full(n, np.nan, dtype=np.float32))

        f.attrs["town"] = args.town
        f.attrs["total_frames"] = n
        f.attrs["total_episodes"] = episode_idx
        f.attrs["good_episodes"] = good_episodes
        f.attrs["total_attempts"] = total_attempts
        f.attrs["collection_type"] = "clean_normal_driving_v5_town06_same_route"
        f.attrs["obstacle_count"] = 0

    print(f"\nSaved {n} frames ({good_episodes} episodes) to {args.output}")


if __name__ == "__main__":
    main()

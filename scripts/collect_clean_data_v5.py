"""Collect no-obstacle clean driving data (v5-compatible HDF5).

Purpose
-------
This script collects obstacle-free normal driving data on the same Town06_Opt
Task-B routes used by collect_obstacle_data_v5.py. The output schema is
compatible with merge_data_v5.py and obstacle v5 datasets.

Key properties
--------------
- No obstacles are spawned.
- task_labels = 0 for all frames.
- obstacle_visible = False for all frames.
- obstacle metadata keys are still written with default values so that this
  file can be merged directly with v5 obstacle datasets.
- BasicAgent follows the same route definitions as obstacle v5 collection.

Example
-------
uv run python scripts/collect_clean_data_v5.py \
    --town Town06_Opt \
    --num_samples 50000 \
    --output data/expert_clean_town06_v5.h5 \
    --target_speed 25.0 \
    --seed 43
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np


def _default_obstacle_meta(image_size: int) -> dict:
    """Return default no-obstacle per-frame metadata."""
    return {
        "obstacle_visible": False,
        "obstacle_distance": float("inf"),
        "obstacle_in_front": False,
        "obstacle_lateral": float("nan"),
        "nearest_obstacle_id": -1,
        "visible_obstacle_id": -1,
        "obstacle_bbox": np.full(4, np.nan, dtype=np.float32),
        "obstacle_bbox_area": 0.0,
        "ego_x": float("nan"),
        "ego_y": float("nan"),
        "ego_yaw": float("nan"),
        "obstacle_x": float("nan"),
        "obstacle_y": float("nan"),
    }


def _fill_ego_meta(meta: dict, vehicle) -> dict:
    """Fill ego pose metadata for debugging/alignment."""
    loc = vehicle.get_location()
    yaw = float(vehicle.get_transform().rotation.yaw)
    meta["ego_x"] = float(loc.x)
    meta["ego_y"] = float(loc.y)
    meta["ego_yaw"] = yaw
    return meta


def _goal_distance(vehicle, goal_loc) -> float:
    loc = vehicle.get_location()
    return math.sqrt((loc.x - goal_loc.x) ** 2 + (loc.y - goal_loc.y) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect no-obstacle clean driving data with v5-compatible keys.",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--route_key", default="Town06_Opt_TaskB")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--output", default="data/expert_clean_town06_v5.h5")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--target_speed", type=float, default=25.0)

    parser.add_argument(
        "--noise_sigma",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help=(
            "Gaussian noise std added to [steer, accel] actions. "
            "Default is [0, 0] for clean/preferred driving."
        ),
    )
    parser.add_argument(
        "--min_episode_frames",
        type=int,
        default=100,
        help="Discard episodes shorter than this many frames.",
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.min_episode_frames <= 0:
        raise ValueError("--min_episode_frames must be positive.")

    try:
        import carla  # noqa: F401
        from agents.navigation.basic_agent import BasicAgent
    except ImportError:
        print("ERROR: carla package and CARLA PythonAPI agents are required.")
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

    routes = EVAL_ROUTES.get(args.route_key, [])
    if not routes:
        env.close()
        raise ValueError(f"No routes found for route_key={args.route_key}")

    client = None
    try:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        carla_map = client.get_world().get_map()

        # v5-compatible accumulators
        all_images: list[np.ndarray] = []
        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_expert_actions: list[np.ndarray] = []
        all_episode_ids: list[int] = []
        all_lateral_devs: list[float] = []
        all_lane_ids: list[int] = []
        all_noise_sigmas: list[list[float]] = []
        all_success_flags: list[bool] = []
        all_task_labels: list[int] = []
        all_target_speeds: list[float] = []

        all_obs_visible: list[bool] = []
        all_obs_distance: list[float] = []
        all_obs_in_front: list[bool] = []
        all_obs_lateral: list[float] = []
        all_nearest_obs_id: list[int] = []
        all_visible_obs_id: list[int] = []
        all_obs_bbox: list[np.ndarray] = []
        all_obs_bbox_area: list[float] = []
        all_ego_x: list[float] = []
        all_ego_y: list[float] = []
        all_ego_yaw: list[float] = []
        all_obs_x: list[float] = []
        all_obs_y: list[float] = []

        collected = 0
        episode_idx = 0
        total_attempts = 0
        clean_episodes = 0

        consecutive_errors = 0
        max_consecutive_errors = 5

        print("=" * 100)
        print("Collect clean driving v5")
        print(f"town: {args.town}")
        print(f"route_key: {args.route_key}")
        print(f"target frames: {args.num_samples}")
        print(f"output: {args.output}")
        print(f"target_speed: {args.target_speed}")
        print(f"noise_sigma: {args.noise_sigma}")

        while collected < args.num_samples:
            total_attempts += 1

            route_idx = int(rng.integers(len(routes)))
            route = routes[route_idx]
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location

            try:
                obs = env.reset(spawn_point=start_spawn)
            except Exception as e:
                consecutive_errors += 1
                print(
                    f"CARLA reset error ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive reset errors; saving partial data.")
                    break
                continue

            consecutive_errors = 0
            env.set_goal(goal_loc)

            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            ep_images: list[np.ndarray] = []
            ep_states: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_expert_actions: list[np.ndarray] = []
            ep_lat_devs: list[float] = []
            ep_lane_ids: list[int] = []
            ep_meta: list[dict] = []

            ep_collisions = 0
            ep_error = False

            for _t in range(args.episode_len):
                if collected + len(ep_images) >= args.num_samples:
                    break

                try:
                    control = agent.run_step()
                    expert_action = carla_to_action(
                        control.steer,
                        control.throttle,
                        control.brake,
                    )
                except Exception as e:
                    print(f"  Agent error at frame {_t}: {e}")
                    ep_error = True
                    break

                noise = rng.normal(0.0, np.asarray(args.noise_sigma, dtype=np.float32))
                action = np.clip(expert_action + noise, -1.0, 1.0)

                try:
                    obs, info = env.step(action)
                except Exception as e:
                    print(f"  Step error at frame {_t}: {e}")
                    ep_error = True
                    break

                img, state = obs

                meta = _fill_ego_meta(_default_obstacle_meta(args.image_size), env._vehicle)

                loc = env._vehicle.get_location()
                waypoint = carla_map.get_waypoint(loc)
                lane_id = waypoint.lane_id if waypoint else -1

                if waypoint:
                    wp_loc = waypoint.transform.location
                    lat_dev = math.sqrt((loc.x - wp_loc.x) ** 2 + (loc.y - wp_loc.y) ** 2)
                else:
                    lat_dev = 0.0

                ep_images.append(img)
                ep_states.append(state)
                ep_actions.append(action)
                ep_expert_actions.append(expert_action)
                ep_lat_devs.append(float(lat_dev))
                ep_lane_ids.append(int(lane_id))
                ep_meta.append(meta)

                if info.get("collision", False):
                    ep_collisions += 1
                    break

                if _goal_distance(env._vehicle, goal_loc) < 15.0 or agent.done():
                    break

            ep_frames = len(ep_images)

            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_ids))
                if ep_lane_ids[i] != ep_lane_ids[i - 1]
                and ep_lane_ids[i] != -1
                and ep_lane_ids[i - 1] != -1
            )

            is_clean = (
                not ep_error
                and ep_collisions == 0
                and ep_frames >= args.min_episode_frames
            )

            if is_clean:
                remaining = args.num_samples - collected
                keep = min(ep_frames, remaining)

                for i in range(keep):
                    m = ep_meta[i]
                    all_images.append(ep_images[i])
                    all_states.append(ep_states[i])
                    all_actions.append(ep_actions[i])
                    all_expert_actions.append(ep_expert_actions[i])
                    all_episode_ids.append(episode_idx)
                    all_lateral_devs.append(ep_lat_devs[i])
                    all_lane_ids.append(ep_lane_ids[i])
                    all_noise_sigmas.append(list(map(float, args.noise_sigma)))
                    all_success_flags.append(True)
                    all_task_labels.append(0)
                    all_target_speeds.append(float(args.target_speed))

                    all_obs_visible.append(False)
                    all_obs_distance.append(float("inf"))
                    all_obs_in_front.append(False)
                    all_obs_lateral.append(float("nan"))
                    all_nearest_obs_id.append(-1)
                    all_visible_obs_id.append(-1)
                    all_obs_bbox.append(np.full(4, np.nan, dtype=np.float32))
                    all_obs_bbox_area.append(0.0)

                    all_ego_x.append(m["ego_x"])
                    all_ego_y.append(m["ego_y"])
                    all_ego_yaw.append(m["ego_yaw"])
                    all_obs_x.append(float("nan"))
                    all_obs_y.append(float("nan"))

                collected += keep
                episode_idx += 1
                clean_episodes += 1

                print(
                    f"  Ep {episode_idx:4d} [CLEAN] | {keep:4d}/{ep_frames:4d}fr "
                    f"lc={lane_changes} | {collected}/{args.num_samples}"
                )
            else:
                reason = (
                    f"col={ep_collisions}"
                    if ep_collisions > 0
                    else "error"
                    if ep_error
                    else f"short={ep_frames}"
                )
                print(f"  Attempt {total_attempts} [SKIP {reason}]")

        if not all_images:
            print("ERROR: No clean frames collected.")
            sys.exit(1)

        n = len(all_images)

        with h5py.File(args.output, "w") as f:
            f.create_dataset("images", data=np.stack(all_images), dtype=np.float32)
            f.create_dataset("states", data=np.stack(all_states), dtype=np.float32)
            f.create_dataset("actions", data=np.stack(all_actions), dtype=np.float32)
            f.create_dataset(
                "expert_actions",
                data=np.stack(all_expert_actions),
                dtype=np.float32,
            )
            f.create_dataset("episode_ids", data=np.array(all_episode_ids, dtype=np.int64))
            f.create_dataset("lateral_devs", data=np.array(all_lateral_devs, dtype=np.float32))
            f.create_dataset("lane_ids", data=np.array(all_lane_ids, dtype=np.int32))
            f.create_dataset("noise_sigmas", data=np.array(all_noise_sigmas, dtype=np.float32))
            f.create_dataset("success_flags", data=np.array(all_success_flags, dtype=bool))
            f.create_dataset("task_labels", data=np.array(all_task_labels, dtype=np.int8))
            f.create_dataset("target_speeds", data=np.array(all_target_speeds, dtype=np.float32))

            # v5 obstacle metadata defaults
            f.create_dataset(
                "obstacle_visible",
                data=np.array(all_obs_visible, dtype=bool),
            )
            f.create_dataset(
                "obstacle_distance",
                data=np.array(all_obs_distance, dtype=np.float32),
            )
            f.create_dataset(
                "obstacle_in_front",
                data=np.array(all_obs_in_front, dtype=bool),
            )
            f.create_dataset(
                "obstacle_lateral",
                data=np.array(all_obs_lateral, dtype=np.float32),
            )
            f.create_dataset(
                "nearest_obstacle_id",
                data=np.array(all_nearest_obs_id, dtype=np.int32),
            )
            f.create_dataset(
                "visible_obstacle_id",
                data=np.array(all_visible_obs_id, dtype=np.int32),
            )
            f.create_dataset(
                "obstacle_bbox",
                data=np.stack(all_obs_bbox).astype(np.float32),
            )
            f.create_dataset(
                "obstacle_bbox_area",
                data=np.array(all_obs_bbox_area, dtype=np.float32),
            )
            f.create_dataset("ego_x", data=np.array(all_ego_x, dtype=np.float32))
            f.create_dataset("ego_y", data=np.array(all_ego_y, dtype=np.float32))
            f.create_dataset("ego_yaw", data=np.array(all_ego_yaw, dtype=np.float32))
            f.create_dataset("obstacle_x", data=np.array(all_obs_x, dtype=np.float32))
            f.create_dataset("obstacle_y", data=np.array(all_obs_y, dtype=np.float32))

            f.attrs["town"] = args.town
            f.attrs["route_key"] = args.route_key
            f.attrs["total_frames"] = int(n)
            f.attrs["total_episodes"] = int(episode_idx)
            f.attrs["clean_episodes"] = int(clean_episodes)
            f.attrs["total_attempts"] = int(total_attempts)
            f.attrs["collection_type"] = "clean_no_obstacle_v5"
            f.attrs["camera_image_size"] = int(args.image_size)
            f.attrs["target_speed"] = float(args.target_speed)
            f.attrs["noise_sigma_steer"] = float(args.noise_sigma[0])
            f.attrs["noise_sigma_accel"] = float(args.noise_sigma[1])
            f.attrs["visible_frames"] = 0

        print(
            f"\nSaved {n} clean frames ({clean_episodes} episodes) to {args.output}"
            "\n  obstacle_visible=True frames: 0"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
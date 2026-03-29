"""Collect expert obstacle-avoidance data using CARLA BehaviorAgent.

Spawns static obstacles on multi-lane highway routes and records the
BehaviorAgent performing lane changes to avoid them. The resulting
HDF5 dataset teaches the world model obstacle dynamics so that EFE
minimization can produce obstacle avoidance without privileged info.

Usage:
    uv run python scripts/collect_obstacle_data.py \
        --town Town06_Opt --num_samples 30000 \
        --output data/expert_obstacle_avoidance.h5
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Collect obstacle-avoidance expert data from CARLA",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--output", default="data/expert_obstacle_avoidance.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=2000)
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_speed", type=float, default=30.0)
    args = parser.parse_args()

    try:
        import carla
        from agents.navigation.behavior_agent import BehaviorAgent
    except ImportError:
        print(
            "ERROR: carla package required. "
            "Install with: pip install carla==0.9.16"
        )
        sys.exit(1)

    from active_inference.data.carla_env import CARLADrivingEnv, carla_to_action
    from active_inference.evaluation.obstacles import (
        spawn_obstacles_on_route,
        destroy_obstacles,
    )
    from active_inference.evaluation.routes import (
        EVAL_ROUTES,
        get_route_waypoints,
    )

    rng = np.random.default_rng(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    env = CARLADrivingEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        image_model_size=args.image_size,
    )

    # Use Task B routes for obstacle placement
    route_key = "Town06_Opt_TaskB"
    routes = EVAL_ROUTES.get(route_key, [])
    if not routes:
        print(f"ERROR: No routes found for {route_key}")
        sys.exit(1)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

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

    episode_idx = 0
    collected = 0
    obstacle_actors = []

    # Obstacle placement fractions — vary per episode for diversity
    FRACTION_SETS = [
        [0.15, 0.45, 0.75],
        [0.20, 0.50, 0.80],
        [0.10, 0.40, 0.70],
        [0.25, 0.55, 0.85],
        [0.15, 0.50, 0.85],
    ]

    try:
        while collected < args.num_samples:
            # Pick a route (prefer routes 1 and 2 — longer, more room)
            route_idx = rng.integers(len(routes))
            route = routes[route_idx]
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location
            route_wps = get_route_waypoints(client, args.town, route)

            # Reset env at route start
            destroy_obstacles(obstacle_actors)
            obs = env.reset(spawn_point=start_spawn)
            img, state = obs
            env.set_goal(goal_loc)

            # Spawn obstacles at varied fractions
            fracs = FRACTION_SETS[episode_idx % len(FRACTION_SETS)]
            obstacle_actors = spawn_obstacles_on_route(
                env._world,
                route_wps,
                num_obstacles=args.num_obstacles,
                fractions=fracs,
            )
            n_spawned = len(obstacle_actors)
            if n_spawned == 0:
                print(f"  Ep {episode_idx}: no obstacles spawned, skipping")
                continue

            obs_positions = [
                (ix, iy) for _, ix, iy in obstacle_actors
            ]
            print(
                f"  Ep {episode_idx}: route {route_idx}, "
                f"{n_spawned} obstacles at fracs {fracs[:n_spawned]}"
            )

            # Use BehaviorAgent for obstacle avoidance behavior
            agent = BehaviorAgent(
                env._vehicle,
                behavior="normal",
            )
            agent.set_destination(goal_loc)

            ep_start = collected
            ep_collisions = 0
            ep_lat_devs = []
            ep_lane_list = []

            for t in range(args.episode_len):
                if collected >= args.num_samples:
                    break

                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake,
                )

                # Light noise for diversity (much less than collect_data.py)
                noise = rng.normal(0, 0.05, size=2)
                noisy_action = np.clip(
                    expert_action + noise, -1.0, 1.0,
                )

                obs, info = env.step(noisy_action)
                img, state = obs

                all_images.append(img)
                all_states.append(state)
                all_actions.append(noisy_action)
                all_expert_actions.append(expert_action)
                all_episode_ids.append(episode_idx)
                all_noise_sigmas.append([0.05, 0.05])

                if info.get("collision", False):
                    ep_collisions += 1

                waypoint = env._world.get_map().get_waypoint(
                    env._vehicle.get_location(),
                )
                lane_id = waypoint.lane_id if waypoint else -1
                loc = env._vehicle.get_location()
                if waypoint:
                    wp_loc = waypoint.transform.location
                    lat_dev = math.sqrt(
                        (loc.x - wp_loc.x) ** 2
                        + (loc.y - wp_loc.y) ** 2
                    )
                else:
                    lat_dev = 0.0

                all_lateral_devs.append(lat_dev)
                all_lane_ids.append(lane_id)
                ep_lat_devs.append(lat_dev)
                ep_lane_list.append(lane_id)
                collected += 1

                # Check goal reached
                goal_dist = math.sqrt(
                    (loc.x - goal_loc.x) ** 2
                    + (loc.y - goal_loc.y) ** 2
                )
                if goal_dist < 15.0:
                    break
                if agent.done():
                    break
                if ep_collisions > 2:
                    break

            ep_end = collected
            ep_frames = ep_end - ep_start

            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_list))
                if ep_lane_list[i] != ep_lane_list[i - 1]
                and ep_lane_list[i] != -1
                and ep_lane_list[i - 1] != -1
            )
            is_success = ep_collisions == 0 and lane_changes > 0
            task_label = 1 if lane_changes > 0 else 0

            for _ in range(ep_frames):
                ep_success_flags.append(is_success)
                ep_task_labels.append(task_label)

            episode_idx += 1
            mean_lat = (
                float(np.mean(ep_lat_devs)) if ep_lat_devs else 0.0
            )
            print(
                f"  Ep {episode_idx:4d} | {ep_frames:4d}fr "
                f"col={ep_collisions} lat={mean_lat:.3f} "
                f"lc={lane_changes} ok={is_success} | "
                f"{collected}/{args.num_samples}"
            )

    finally:
        destroy_obstacles(obstacle_actors)
        env.close()

    if not all_images:
        print("ERROR: No frames collected")
        sys.exit(1)

    with h5py.File(args.output, "w") as f:
        f.create_dataset(
            "images", data=np.stack(all_images), dtype=np.float32,
        )
        f.create_dataset(
            "states", data=np.stack(all_states), dtype=np.float32,
        )
        f.create_dataset(
            "actions", data=np.stack(all_actions), dtype=np.float32,
        )
        f.create_dataset(
            "expert_actions",
            data=np.stack(all_expert_actions),
            dtype=np.float32,
        )
        f.create_dataset(
            "episode_ids",
            data=np.array(all_episode_ids, dtype=np.int64),
        )
        f.create_dataset(
            "lateral_devs",
            data=np.array(all_lateral_devs, dtype=np.float32),
        )
        f.create_dataset(
            "lane_ids",
            data=np.array(all_lane_ids, dtype=np.int32),
        )
        f.create_dataset(
            "noise_sigmas",
            data=np.array(all_noise_sigmas, dtype=np.float32),
        )
        f.create_dataset(
            "success_flags",
            data=np.array(ep_success_flags, dtype=bool),
        )
        f.create_dataset(
            "task_labels",
            data=np.array(ep_task_labels, dtype=np.int8),
        )

        f.attrs["town"] = args.town
        f.attrs["total_frames"] = collected
        f.attrs["total_episodes"] = episode_idx
        f.attrs["num_obstacles_per_episode"] = args.num_obstacles
        f.attrs["collection_type"] = "obstacle_avoidance"

    successes = sum(1 for s in ep_success_flags if s)
    lc_frames = sum(1 for t in ep_task_labels if t == 1)
    print(
        f"\nSaved {collected} frames ({episode_idx} episodes) "
        f"to {args.output}"
    )
    print(
        f"Success frames: {successes}/{collected} "
        f"({100 * successes / max(collected, 1):.1f}%)"
    )
    print(f"Lane-change frames: {lc_frames}/{collected}")


if __name__ == "__main__":
    main()

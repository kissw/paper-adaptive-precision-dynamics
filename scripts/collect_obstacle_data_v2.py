"""Collect clean obstacle-avoidance demonstrations via scripted lane changes.

Unlike v1 (BehaviorAgent, 0% success), this script uses CARLA's BasicAgent
for straight driving and forces a scripted lane change when approaching
each obstacle. This produces clean demonstrations: approach → lane change
→ pass obstacle → return to lane.

The key insight: the world model needs to see the VISUAL transition of
"obstacle in view → steer → obstacle exits view" to learn the association.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Collect scripted obstacle-avoidance data",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--output", default="data/expert_obstacle_avoidance_v2.h5")
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
        from agents.navigation.basic_agent import BasicAgent
    except ImportError:
        print("ERROR: carla package required")
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

    route_key = "Town06_Opt_TaskB"
    routes = EVAL_ROUTES.get(route_key, [])
    if not routes:
        print(f"ERROR: No routes for {route_key}")
        sys.exit(1)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    carla_map = client.get_world().get_map()

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

    FRACTION_SETS = [
        [0.15, 0.45, 0.75],
        [0.20, 0.50, 0.80],
        [0.10, 0.40, 0.70],
        [0.25, 0.55, 0.85],
        [0.15, 0.50, 0.85],
    ]

    # Scripted lane-change parameters
    DETECT_DIST = 30.0    # start lane change at this distance
    STEER_MAG = 0.4       # steering magnitude during lane change
    STEER_FRAMES = 20     # frames of active steering
    COAST_FRAMES = 15     # frames of coasting in new lane
    RETURN_FRAMES = 20    # frames of counter-steering to return
    EVASION_DIRECTIONS = [-1.0, 1.0]  # left, right — alternate

    try:
        while collected < args.num_samples:
            route_idx = rng.integers(len(routes))
            route = routes[route_idx]
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location
            route_wps = get_route_waypoints(client, args.town, route)

            destroy_obstacles(obstacle_actors)
            obs = env.reset(spawn_point=start_spawn)
            img, state = obs
            env.set_goal(goal_loc)

            fracs = FRACTION_SETS[episode_idx % len(FRACTION_SETS)]
            obstacle_actors = spawn_obstacles_on_route(
                env._world, route_wps,
                num_obstacles=args.num_obstacles, fractions=fracs,
            )
            n_spawned = len(obstacle_actors)
            if n_spawned == 0:
                continue

            obs_positions = [(ix, iy) for _, ix, iy in obstacle_actors]
            print(
                f"  Ep {episode_idx}: route {route_idx}, "
                f"{n_spawned} obstacles"
            )

            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            ep_start = collected
            ep_collisions = 0
            ep_lat_devs = []
            ep_lane_list = []

            # Scripted lane-change state machine
            lc_active = False
            lc_phase = 0       # 0=steer, 1=coast, 2=return
            lc_counter = 0
            lc_direction = 0.0
            lc_obs_idx = 0     # which obstacle we're avoiding next
            obstacles_cleared = [False] * n_spawned

            for t in range(args.episode_len):
                if collected >= args.num_samples:
                    break

                # Get autopilot control
                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake,
                )

                loc = env._vehicle.get_location()

                # Check distance to next uncleared obstacle
                if not lc_active and lc_obs_idx < n_spawned:
                    ox, oy = obs_positions[lc_obs_idx]
                    dist = math.sqrt(
                        (loc.x - ox) ** 2 + (loc.y - oy) ** 2
                    )
                    # Check if obstacle is ahead (forward projection)
                    yaw = math.radians(
                        env._vehicle.get_transform().rotation.yaw
                    )
                    dx, dy = ox - loc.x, oy - loc.y
                    fwd_proj = dx * math.cos(yaw) + dy * math.sin(yaw)

                    if fwd_proj > 0 and dist < DETECT_DIST:
                        # Start lane change
                        lc_active = True
                        lc_phase = 0
                        lc_counter = 0
                        # Alternate direction, or use lane geometry
                        wp = carla_map.get_waypoint(loc)
                        if wp:
                            left = wp.get_left_lane()
                            right = wp.get_right_lane()
                            has_left = (
                                left is not None
                                and str(left.lane_type) == "Driving"
                            )
                            has_right = (
                                right is not None
                                and str(right.lane_type) == "Driving"
                            )
                            if has_left and not has_right:
                                lc_direction = -1.0
                            elif has_right and not has_left:
                                lc_direction = 1.0
                            else:
                                lc_direction = EVASION_DIRECTIONS[
                                    lc_obs_idx % 2
                                ]
                        else:
                            lc_direction = -1.0
                        print(
                            f"    t={t} LC start: obs[{lc_obs_idx}] "
                            f"dist={dist:.1f}m dir={lc_direction:+.0f}"
                        )

                    # Mark obstacle as cleared if behind
                    if fwd_proj < -10.0 and not obstacles_cleared[lc_obs_idx]:
                        obstacles_cleared[lc_obs_idx] = True
                        lc_obs_idx = min(lc_obs_idx + 1, n_spawned - 1)

                # Apply scripted lane-change override
                if lc_active:
                    action = expert_action.copy()
                    if lc_phase == 0:  # steer into adjacent lane
                        action[0] = lc_direction * STEER_MAG
                        lc_counter += 1
                        if lc_counter >= STEER_FRAMES:
                            lc_phase = 1
                            lc_counter = 0
                    elif lc_phase == 1:  # coast in adjacent lane
                        action[0] = 0.0
                        lc_counter += 1
                        if lc_counter >= COAST_FRAMES:
                            lc_phase = 2
                            lc_counter = 0
                    elif lc_phase == 2:  # return to original lane
                        action[0] = -lc_direction * STEER_MAG * 0.8
                        lc_counter += 1
                        if lc_counter >= RETURN_FRAMES:
                            lc_active = False
                            lc_phase = 0
                            lc_counter = 0
                    # Keep throttle from autopilot
                    action[1] = expert_action[1]
                    noisy_action = np.clip(action, -1.0, 1.0)
                else:
                    # Normal driving with light noise
                    noise = rng.normal(0, 0.03, size=2)
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
                all_noise_sigmas.append([0.03, 0.03])

                if info.get("collision", False):
                    ep_collisions += 1

                waypoint = carla_map.get_waypoint(loc)
                lane_id = waypoint.lane_id if waypoint else -1
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

                goal_dist = math.sqrt(
                    (loc.x - goal_loc.x) ** 2
                    + (loc.y - goal_loc.y) ** 2
                )
                if goal_dist < 15.0:
                    break
                if agent.done():
                    break
                if ep_collisions > 5:
                    break

            ep_frames = collected - ep_start
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
            mean_lat = float(np.mean(ep_lat_devs)) if ep_lat_devs else 0.0
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
            data=np.stack(all_expert_actions), dtype=np.float32,
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
        f.attrs["collection_type"] = "scripted_obstacle_avoidance_v2"

    successes = sum(1 for s in ep_success_flags if s)
    lc_frames = sum(1 for t in ep_task_labels if t == 1)
    print(
        f"\nSaved {collected} frames ({episode_idx} episodes) "
        f"to {args.output}"
    )
    print(
        f"Success: {successes}/{collected} "
        f"({100 * successes / max(collected, 1):.1f}%)"
    )
    print(f"Lane-change frames: {lc_frames}/{collected}")


if __name__ == "__main__":
    main()

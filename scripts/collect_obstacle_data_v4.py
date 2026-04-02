"""Collect clean obstacle-avoidance data via waypoint-based lane change (v4).

Instead of fighting the autopilot with scripted steering, this version
redirects the BasicAgent to a waypoint in the adjacent lane when near
an obstacle. The autopilot's own path planner handles the lane change
smoothly and collision-free.

Key insight: the autopilot needs a new DESTINATION in the adjacent lane,
not a steering override. The autopilot's internal planner will generate
a smooth lane-change trajectory to reach the new waypoint.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Collect clean obstacle-avoidance data (v4)",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--output", default="data/expert_obstacle_v4.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=3000)
    parser.add_argument("--num_obstacles", type=int, default=2)
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
    clean_episodes = 0
    total_attempts = 0

    FRACTION_SETS = [
        [0.30, 0.70],
        [0.25, 0.65],
        [0.35, 0.75],
        [0.20, 0.60],
        [0.40, 0.80],
    ]

    DETECT_DIST = 60.0  # redirect agent at 60m

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

            fracs = FRACTION_SETS[total_attempts % len(FRACTION_SETS)]
            obstacle_actors = spawn_obstacles_on_route(
                env._world, route_wps,
                num_obstacles=args.num_obstacles, fractions=fracs,
            )
            n_spawned = len(obstacle_actors)
            if n_spawned == 0:
                total_attempts += 1
                continue

            obs_positions = [(ix, iy) for _, ix, iy in obstacle_actors]

            # Create agent heading to the final goal
            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            ep_images = []
            ep_states = []
            ep_actions = []
            ep_expert_actions = []
            ep_lat_devs = []
            ep_lane_list = []
            ep_collisions = 0

            current_obs_idx = 0
            passed_obstacles = set()
            redirect_done = set()  # track which obstacles triggered redirect

            total_attempts += 1

            for t in range(args.episode_len):
                loc = env._vehicle.get_location()
                yaw = math.radians(
                    env._vehicle.get_transform().rotation.yaw,
                )

                # Track passed obstacles
                for oi, (ox, oy) in enumerate(obs_positions):
                    if oi in passed_obstacles:
                        continue
                    dx, dy = ox - loc.x, oy - loc.y
                    fwd_proj = dx * math.cos(yaw) + dy * math.sin(yaw)
                    if fwd_proj < -15.0:
                        passed_obstacles.add(oi)
                        if oi == current_obs_idx:
                            current_obs_idx += 1
                            # After passing obstacle, redirect back
                            # to final goal so agent returns to route
                            agent.set_destination(goal_loc)

                # Redirect agent to adjacent lane waypoint when near
                if (current_obs_idx < n_spawned
                        and current_obs_idx not in passed_obstacles
                        and current_obs_idx not in redirect_done):
                    ox, oy = obs_positions[current_obs_idx]
                    dx, dy = ox - loc.x, oy - loc.y
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    fwd_proj = dx * math.cos(yaw) + dy * math.sin(yaw)

                    if fwd_proj > 0 and dist < DETECT_DIST:
                        # Find waypoint past obstacle in adjacent lane
                        obs_wp = carla_map.get_waypoint(
                            carla.Location(x=ox, y=oy),
                        )
                        if obs_wp:
                            # Get adjacent lane
                            left = obs_wp.get_left_lane()
                            right = obs_wp.get_right_lane()
                            adj = None
                            if (left and str(left.lane_type) == "Driving"):
                                adj = left
                            elif (right
                                  and str(right.lane_type) == "Driving"):
                                adj = right

                            if adj:
                                # Get a waypoint 30m past obstacle
                                # in the adjacent lane
                                ahead = adj.next(30.0)
                                if ahead:
                                    target = ahead[0].transform.location
                                    agent.set_destination(target)
                                    redirect_done.add(current_obs_idx)
                                    print(
                                        f"    t={t} REDIRECT obs["
                                        f"{current_obs_idx}] "
                                        f"dist={dist:.0f}m → "
                                        f"adj lane"
                                    )

                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake,
                )

                # Very light noise
                noise = rng.normal(0, 0.02, size=2)
                noisy_action = np.clip(
                    expert_action + noise, -1.0, 1.0,
                )

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
                        (loc.x - wp_loc.x) ** 2
                        + (loc.y - wp_loc.y) ** 2
                    )
                else:
                    lat_dev = 0.0
                ep_lat_devs.append(lat_dev)
                ep_lane_list.append(lane_id)

                goal_dist = math.sqrt(
                    (loc.x - goal_loc.x) ** 2
                    + (loc.y - goal_loc.y) ** 2
                )
                if goal_dist < 15.0 or agent.done():
                    break
                if ep_collisions > 0:
                    break

            ep_frames = len(ep_images)
            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_list))
                if ep_lane_list[i] != ep_lane_list[i - 1]
                and ep_lane_list[i] != -1
                and ep_lane_list[i - 1] != -1
            )
            is_clean = ep_collisions == 0 and lane_changes > 0

            if is_clean and ep_frames > 100:
                for i in range(ep_frames):
                    all_images.append(ep_images[i])
                    all_states.append(ep_states[i])
                    all_actions.append(ep_actions[i])
                    all_expert_actions.append(ep_expert_actions[i])
                    all_episode_ids.append(episode_idx)
                    all_lateral_devs.append(ep_lat_devs[i])
                    all_lane_ids.append(ep_lane_list[i])
                    all_noise_sigmas.append([0.02, 0.02])
                    ep_success_flags.append(True)
                    ep_task_labels.append(1)
                collected += ep_frames
                episode_idx += 1
                clean_episodes += 1
                print(
                    f"  Ep {episode_idx:4d} [CLEAN] | {ep_frames:4d}fr "
                    f"lc={lane_changes} | "
                    f"{collected}/{args.num_samples} "
                    f"({clean_episodes}/{total_attempts} clean)"
                )
            else:
                reason = (
                    f"col={ep_collisions}" if ep_collisions > 0
                    else f"no_lc" if lane_changes == 0
                    else f"short={ep_frames}"
                )
                print(
                    f"  Attempt {total_attempts} [SKIP {reason}] | "
                    f"{ep_frames}fr lc={lane_changes}"
                )

            if collected >= args.num_samples:
                break

    finally:
        destroy_obstacles(obstacle_actors)
        env.close()

    if not all_images:
        print("ERROR: No clean frames collected")
        sys.exit(1)

    n = min(len(all_images), args.num_samples)
    with h5py.File(args.output, "w") as f:
        f.create_dataset(
            "images", data=np.stack(all_images[:n]), dtype=np.float32,
        )
        f.create_dataset(
            "states", data=np.stack(all_states[:n]), dtype=np.float32,
        )
        f.create_dataset(
            "actions", data=np.stack(all_actions[:n]), dtype=np.float32,
        )
        f.create_dataset(
            "expert_actions",
            data=np.stack(all_expert_actions[:n]), dtype=np.float32,
        )
        f.create_dataset(
            "episode_ids",
            data=np.array(all_episode_ids[:n], dtype=np.int64),
        )
        f.create_dataset(
            "lateral_devs",
            data=np.array(all_lateral_devs[:n], dtype=np.float32),
        )
        f.create_dataset(
            "lane_ids",
            data=np.array(all_lane_ids[:n], dtype=np.int32),
        )
        f.create_dataset(
            "noise_sigmas",
            data=np.array(all_noise_sigmas[:n], dtype=np.float32),
        )
        f.create_dataset(
            "success_flags",
            data=np.array(ep_success_flags[:n], dtype=bool),
        )
        f.create_dataset(
            "task_labels",
            data=np.array(ep_task_labels[:n], dtype=np.int8),
        )

        f.attrs["town"] = args.town
        f.attrs["total_frames"] = n
        f.attrs["total_episodes"] = episode_idx
        f.attrs["clean_episodes"] = clean_episodes
        f.attrs["total_attempts"] = total_attempts
        f.attrs["collection_type"] = "waypoint_obstacle_avoidance_v4"

    print(
        f"\nSaved {n} clean frames ({clean_episodes} episodes, "
        f"{total_attempts} attempts) to {args.output}"
    )
    print(
        f"Clean rate: {clean_episodes}/{total_attempts} "
        f"({100 * clean_episodes / max(total_attempts, 1):.0f}%)"
    )


if __name__ == "__main__":
    main()

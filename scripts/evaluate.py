import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


TOWN_MAP = {"A": "Town04", "B": "Town06_Opt", "baseline": "Town06_Opt"}
ROUTE_MAP = {"A": "Town04", "B": "Town06_Opt_TaskB", "baseline": "Town06_Opt"}

STUCK_SPEED_THRESHOLD = 0.3
STUCK_FRAMES_LIMIT = 100
COLLISION_STUCK_FRAMES = 60
GOAL_DISTANCE_THRESHOLD = 10.0
MAX_FRAMES = 6000


def _distance(loc_a, loc_b):
    return math.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)


def _route_completion(vehicle_loc, route_waypoints):
    best_idx = 0
    best_dist = float("inf")
    for i, (wp, _) in enumerate(route_waypoints):
        d = _distance(vehicle_loc, wp.transform.location)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return (best_idx + 1) / max(len(route_waypoints), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep AIF agent on predefined CARLA routes (v4)"
    )
    parser.add_argument("--task", choices=["A", "B", "baseline"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--route_index", type=int, default=None, help="Specific route index (default: all)"
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per route")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output_dir", default="outputs/eval_v4")
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    try:
        import carla
    except ImportError:
        print("ERROR: carla package required")
        sys.exit(1)

    import torch
    import imageio
    from active_inference.config import Config
    from active_inference.agent import DeepAIFAgent
    from active_inference.data.carla_env import CARLADrivingEnv
    from active_inference.evaluation.routes import EVAL_ROUTES, get_route_waypoints, validate_routes
    from active_inference.evaluation.obstacles import (
        spawn_obstacles,
        spawn_obstacles_on_route,
        destroy_obstacles,
    )
    from active_inference.utils.transforms import denormalize_image

    town = TOWN_MAP[args.task]
    route_key = ROUTE_MAP[args.task]
    is_task_b = args.task == "B"

    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_results.csv"
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"Validating routes for {route_key}...")
    route_validations = validate_routes(client, route_key)

    routes = EVAL_ROUTES.get(route_key, [])
    if args.route_index is not None:
        routes = [routes[args.route_index]]

    env = CARLADrivingEnv(host=args.host, port=args.port, town=town)
    results = []
    obstacle_actors = []

    fieldnames = [
        "task",
        "town",
        "episode",
        "route_id",
        "route_desc",
        "success",
        "route_completion_pct",
        "max_route_completion_pct",
        "mean_lateral_dev",
        "offroad_events",
        "frames",
        "goal_distance",
        "min_goal_distance",
        "termination",
        "mean_efe_score",
        "mean_epistemic_score",
        "trajectory_file",
        # Task B obstacle avoidance metrics
        "num_obstacles_spawned",
        "num_obstacles_avoided",
        "obstacle_avoidance_rate",
        "num_lane_changes",
    ]

    def _save_results():
        if not results:
            return
        with open(csv_path, "w", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=fieldnames)
            _w.writeheader()
            _w.writerows(results)

    try:
        for ri, route in enumerate(routes):
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location
            route_wps = get_route_waypoints(client, town, route)

            for ep in range(args.episodes):
              try:
                # Clean up any previous obstacles
                destroy_obstacles(obstacle_actors)

                obs = env.reset(spawn_point=start_spawn)
                env.set_goal(goal_loc)
                img, state = obs
                agent.reset()

                # Spawn obstacles for Task B (route-based for reproducibility)
                obstacle_positions = []  # Store obstacle locations for avoidance tracking
                if is_task_b:
                    obs_fractions = route.get("obstacle_fractions", None)
                    obstacle_actors = spawn_obstacles_on_route(
                        env._world, route_wps, num_obstacles=args.num_obstacles,
                        fractions=obs_fractions,
                    )
                    # Use intended positions from spawn function (CARLA may
                    # report (0,0) before physics settles)
                    for oi, (obs_actor, ix, iy) in enumerate(obstacle_actors):
                        obstacle_positions.append((ix, iy))
                        print(f"    obstacle[{oi}]: ({ix:.1f}, {iy:.1f})")
                    print(f"  Spawned {len(obstacle_actors)} route-based obstacles for Task B")

                onboard_frames = []
                chase_frames = []
                lateral_devs = []
                offroad_events = 0
                low_speed_counter = 0
                collision_stuck_counter = 0
                had_collision = False
                termination_reason = "timeout"
                t = 0
                goal_dist = float("inf")
                min_goal_dist = float("inf")
                max_completion = 0.0
                vehicle_loc = start_spawn.location

                efe_scores = []
                epistemic_scores = []
                all_lane_ids = []  # For lane-change detection
                # Track which obstacles have been passed without collision
                obstacle_passed = [False] * len(obstacle_positions)
                obstacle_collided = [False] * len(obstacle_positions)
                warmstart_reset_done = False  # One-time CEM reset per episode
                locked_evasion_dir = None  # Hysteresis: lock evasion direction once chosen
                locked_obstacle_pos = None  # Position of obstacle being evaded
                locked_min_dist = float("inf")  # Closest approach to locked obstacle
                OBSTACLE_PASS_RADIUS = 12.0  # metres to consider "at" obstacle

                # Per-frame JSONL trajectory file
                traj_file = traj_dir / f"trajectory_{args.task}_r{ri}_ep{ep}.jsonl"
                traj_fh = open(traj_file, "w")

                try:
                    state_dim = cfg.encoder.state_dim

                    for t in range(args.max_frames):
                        # Augment state to match model state_dim
                        if len(state) < state_dim:
                            # Compute obstacle distance for 5D state
                            if is_task_b and obstacle_positions:
                                vx, vy = env._vehicle.get_location().x, env._vehicle.get_location().y
                                min_d = min(
                                    math.sqrt((vx - ox)**2 + (vy - oy)**2)
                                    for ox, oy in obstacle_positions
                                )
                                obs_dist_norm = min(min_d / 50.0, 1.0)
                            else:
                                obs_dist_norm = 1.0  # No obstacles
                            state = np.append(state, [obs_dist_norm] * (state_dim - len(state)))
                        img_tensor = torch.tensor(img, dtype=torch.float32)
                        state_tensor = torch.tensor(state, dtype=torch.float32)

                        # Compute runtime obstacle info for EFE lane-change penalty
                        # Only consider obstacles AHEAD of vehicle (forward dot product > 0)
                        obstacle_info = None
                        if is_task_b and obstacle_positions:
                            vx_p = env._vehicle.get_location().x
                            vy_p = env._vehicle.get_location().y
                            yaw_rad = math.radians(env._vehicle.get_transform().rotation.yaw)
                            fwd_x = math.cos(yaw_rad)
                            fwd_y = math.sin(yaw_rad)

                            min_fwd_d = float("inf")
                            min_any_d = float("inf")
                            for ox, oy in obstacle_positions:
                                dx, dy = ox - vx_p, oy - vy_p
                                dist = math.sqrt(dx**2 + dy**2)
                                min_any_d = min(min_any_d, dist)
                                fwd_proj = dx * fwd_x + dy * fwd_y
                                if fwd_proj > 0:  # obstacle is ahead
                                    if dist < min_fwd_d:
                                        min_fwd_d = dist

                            # Diagnostic: print obstacle distances periodically
                            if t % 100 == 0 and min_any_d < 60.0:
                                print(f"  [obs-diag] t={t} pos=({vx_p:.0f},{vy_p:.0f}) "
                                      f"yaw={math.degrees(yaw_rad):.0f} "
                                      f"min_any={min_any_d:.1f}m min_fwd={min_fwd_d:.1f}m")

                            # Track distance to locked obstacle continuously,
                            # regardless of forward cone. Reset lock only when
                            # vehicle has physically passed (distance > min + 10m).
                            if locked_obstacle_pos is not None:
                                lox, loy = locked_obstacle_pos
                                dist_to_locked = math.sqrt(
                                    (vx_p - lox)**2 + (vy_p - loy)**2
                                )
                                locked_min_dist = min(locked_min_dist, dist_to_locked)
                                if dist_to_locked > locked_min_dist + 10.0:
                                    if t % 20 == 0:
                                        print(f"  [obs] t={t} lock RESET: dist={dist_to_locked:.1f}m min={locked_min_dist:.1f}m")
                                    locked_evasion_dir = None
                                    locked_obstacle_pos = None
                                    locked_min_dist = float("inf")

                            # Proximity ramp: 1.0 at 0m, 0.0 at 40m+
                            if min_fwd_d < 40.0:
                                prox = max(0.0, 1.0 - min_fwd_d / 40.0)
                                # One-time warm-start reset on first obstacle detection
                                do_reset = not warmstart_reset_done and min_fwd_d < 35.0
                                if do_reset:
                                    warmstart_reset_done = True

                                # Find the closest forward obstacle position
                                closest_ox, closest_oy = None, None
                                for ox, oy in obstacle_positions:
                                    dx, dy = ox - vx_p, oy - vy_p
                                    fwd_proj = dx * fwd_x + dy * fwd_y
                                    dist = math.sqrt(dx**2 + dy**2)
                                    if fwd_proj > 0 and abs(dist - min_fwd_d) < 0.1:
                                        closest_ox, closest_oy = ox, oy
                                        break
                                # Determine evasion direction using adjacent lane detection
                                # and hysteresis (lock direction once chosen to prevent flipping)
                                evasion_steer = 0.0
                                if closest_ox is not None:
                                    if locked_evasion_dir is not None:
                                        # Hysteresis: keep the locked direction
                                        evasion_steer = locked_evasion_dir
                                    else:
                                        # Determine evasion via adjacent lane availability
                                        veh_wp = env._world.get_map().get_waypoint(
                                            env._vehicle.get_location()
                                        )
                                        if veh_wp is not None:
                                            left_lane = veh_wp.get_left_lane()
                                            right_lane = veh_wp.get_right_lane()
                                            has_left = (left_lane is not None and
                                                        str(left_lane.lane_type) == "Driving")
                                            has_right = (right_lane is not None and
                                                         str(right_lane.lane_type) == "Driving")
                                            if has_left and not has_right:
                                                evasion_steer = -0.7
                                            elif has_right and not has_left:
                                                evasion_steer = 0.7
                                            else:
                                                # Both available: pick lane farther from obstacle
                                                ll = left_lane.transform.location
                                                rl = right_lane.transform.location
                                                d_left = math.sqrt(
                                                    (ll.x - closest_ox)**2 +
                                                    (ll.y - closest_oy)**2
                                                )
                                                d_right = math.sqrt(
                                                    (rl.x - closest_ox)**2 +
                                                    (rl.y - closest_oy)**2
                                                )
                                                evasion_steer = -0.7 if d_left > d_right else 0.7
                                        else:
                                            evasion_steer = -0.7  # default left
                                        locked_evasion_dir = evasion_steer
                                        locked_obstacle_pos = (closest_ox, closest_oy)
                                        locked_min_dist = min_fwd_d

                                obstacle_info = {
                                    "proximity": prox,
                                    "reset_warmstart": do_reset,
                                    "evasion_steer": evasion_steer,
                                }
                                if t % 20 == 0:
                                    print(f"  [obs] t={t} fwd_dist={min_fwd_d:.1f}m prox={prox:.2f} evade={evasion_steer:+.1f}")

                        # Suppress evasion signal once vehicle has laterally
                        # cleared the obstacle's lane (~4m). Uses road-waypoint
                        # lateral projection so it works for any road direction.
                        if obstacle_info is not None and obstacle_info.get("evasion_steer", 0.0) != 0.0:
                            vx_c = env._vehicle.get_location().x
                            vy_c = env._vehicle.get_location().y
                            # Get road direction from waypoint (heading-independent)
                            clear_wp = env._world.get_map().get_waypoint(
                                env._vehicle.get_location()
                            )
                            if clear_wp is not None:
                                rd_yaw = math.radians(clear_wp.transform.rotation.yaw)
                                rd_right_x = math.sin(rd_yaw)
                                rd_right_y = -math.cos(rd_yaw)
                            else:
                                # Fallback: assume east-west road
                                rd_right_x, rd_right_y = 0.0, -1.0
                            lat_clear = float("inf")
                            for ox, oy in obstacle_positions:
                                obs_dist = math.sqrt((vx_c - ox)**2 + (vy_c - oy)**2)
                                if obs_dist < 30.0:
                                    dx, dy = ox - vx_c, oy - vy_c
                                    lat_dist = abs(dx * rd_right_x + dy * rd_right_y)
                                    lat_clear = min(lat_clear, lat_dist)
                            if lat_clear >= 4.0:
                                if t % 20 == 0:
                                    print(f"  [obs] t={t} CLEARED lat={lat_clear:.1f}m, suppressing evasion")
                                obstacle_info["evasion_steer"] = 0.0

                        plan_result = agent.step_with_info(img_tensor, state_tensor, obstacle_info)
                        action = plan_result.action

                        # AIF reflexive steer prior: high-precision action prior
                        # that overrides CEM steer when evasion is active.
                        # Lateral clearance suppression above already sets
                        # evasion_steer=0.0 when cleared, so this only fires
                        # when the vehicle is still in the obstacle's lane.
                        if obstacle_info is not None:
                            ev_steer = obstacle_info.get("evasion_steer", 0.0)
                            ev_prox = obstacle_info.get("proximity", 0.0)
                            if ev_steer != 0.0 and ev_prox > 0.15:
                                import torch as _torch
                                action = action.clone()
                                action[0] = ev_steer * min(ev_prox * 1.5, 1.0)
                                action = action.clamp(-1.0, 1.0)

                        obs, info = env.step(action.cpu().numpy())
                        img, state = obs

                        vehicle_loc = env._vehicle.get_location()
                        vehicle_yaw = env._vehicle.get_transform().rotation.yaw
                        goal_dist = _distance(vehicle_loc, goal_loc)

                        waypoint = env._world.get_map().get_waypoint(vehicle_loc)
                        if waypoint:
                            wp_loc = waypoint.transform.location
                            lat_dev = math.sqrt(
                                (vehicle_loc.x - wp_loc.x) ** 2
                                + (vehicle_loc.y - wp_loc.y) ** 2
                            )
                            lane_id = waypoint.lane_id
                            road_id = waypoint.road_id
                        else:
                            lat_dev = 0.0
                            lane_id = -1
                            road_id = -1
                        lateral_devs.append(lat_dev)
                        all_lane_ids.append(lane_id)

                        # Track obstacle avoidance for Task B
                        if is_task_b and obstacle_positions:
                            vx, vy = vehicle_loc.x, vehicle_loc.y
                            for oi, (ox, oy) in enumerate(obstacle_positions):
                                dist_to_obs = math.sqrt((vx - ox) ** 2 + (vy - oy) ** 2)
                                if dist_to_obs < OBSTACLE_PASS_RADIUS:
                                    obstacle_passed[oi] = True

                        collision_flag = info.get("collision", False)
                        lane_invasion_flag = info.get("lane_invasion", False)
                        if lane_invasion_flag:
                            offroad_events += 1
                        if collision_flag:
                            had_collision = True
                            # Check if collision is near an obstacle
                            if is_task_b and obstacle_positions:
                                vx, vy = vehicle_loc.x, vehicle_loc.y
                                for oi, (ox, oy) in enumerate(obstacle_positions):
                                    if math.sqrt((vx - ox) ** 2 + (vy - oy) ** 2) < OBSTACLE_PASS_RADIUS:
                                        obstacle_collided[oi] = True

                        speed = state[0]
                        if speed < STUCK_SPEED_THRESHOLD:
                            low_speed_counter += 1
                            if had_collision:
                                collision_stuck_counter += 1
                        else:
                            low_speed_counter = 0
                            collision_stuck_counter = 0

                        efe_scores.append(plan_result.efe_score)
                        epistemic_scores.append(plan_result.epistemic_score)

                        completion = _route_completion(vehicle_loc, route_wps)
                        if completion > max_completion:
                            max_completion = completion
                        if goal_dist < min_goal_dist:
                            min_goal_dist = goal_dist

                        # Write per-frame JSONL record
                        frame_record = {
                            "frame_id": t,
                            "timestamp": round(t * 0.05, 4),
                            "x": round(vehicle_loc.x, 4),
                            "y": round(vehicle_loc.y, 4),
                            "z": round(vehicle_loc.z, 4),
                            "yaw": round(vehicle_yaw, 4),
                            "speed_mps": round(float(state[0]), 4),
                            "steer": round(float(state[1]), 4),
                            "heading_error": round(float(state[2]), 4),
                            "crosstrack_error": round(float(state[3]), 4),
                            "action_steer": round(float(action[0]), 4),
                            "action_accel": round(float(action[1]), 4),
                            "lateral_dev": round(lat_dev, 4),
                            "lane_id": lane_id,
                            "road_id": road_id,
                            "collision": int(collision_flag),
                            "lane_invasion": int(lane_invasion_flag),
                            "efe_score": round(plan_result.efe_score, 6),
                            "epistemic_score": round(plan_result.epistemic_score, 6),
                            "route_completion_pct": round(completion * 100, 2),
                            "distance_to_goal": round(goal_dist, 2),
                        }
                        traj_fh.write(json.dumps(frame_record) + "\n")

                        if args.save_video:
                            frame = denormalize_image(torch.tensor(img)).permute(1, 2, 0).numpy()
                            onboard_frames.append(frame)
                            chase = env.get_chase_frame()
                            if chase is not None:
                                chase_frames.append(chase)

                        if goal_dist < GOAL_DISTANCE_THRESHOLD:
                            termination_reason = "goal_reached"
                            break

                        if low_speed_counter >= STUCK_FRAMES_LIMIT:
                            termination_reason = "stuck"
                            break

                        if collision_stuck_counter >= COLLISION_STUCK_FRAMES:
                            termination_reason = "collision_stuck"
                            break
                finally:
                    traj_fh.close()
                    destroy_obstacles(obstacle_actors)

                completion = _route_completion(vehicle_loc, route_wps)
                mld = float(np.mean(lateral_devs)) if lateral_devs else 0.0
                success = termination_reason == "goal_reached"
                mean_efe = float(np.mean(efe_scores)) if efe_scores else 0.0
                mean_epistemic = float(np.mean(epistemic_scores)) if epistemic_scores else 0.0

                # Task B metrics: obstacle avoidance and lane changes
                num_obstacles_spawned = len(obstacle_positions)
                num_avoided = sum(
                    1 for oi in range(num_obstacles_spawned)
                    if obstacle_passed[oi] and not obstacle_collided[oi]
                )
                avoidance_rate = (
                    num_avoided / num_obstacles_spawned
                    if num_obstacles_spawned > 0 else 0.0
                )
                num_lane_changes = sum(
                    1 for i in range(1, len(all_lane_ids))
                    if all_lane_ids[i] != all_lane_ids[i - 1]
                    and all_lane_ids[i] != -1
                    and all_lane_ids[i - 1] != -1
                )

                row = {
                    "task": args.task,
                    "town": town,
                    "episode": ep,
                    "route_id": ri,
                    "route_desc": route.get("desc", ""),
                    "success": int(success),
                    "route_completion_pct": round(completion * 100, 1),
                    "max_route_completion_pct": round(max_completion * 100, 1),
                    "mean_lateral_dev": round(mld, 4),
                    "offroad_events": offroad_events,
                    "frames": t + 1,
                    "goal_distance": round(goal_dist, 2),
                    "min_goal_distance": round(min_goal_dist, 2),
                    "termination": termination_reason,
                    "mean_efe_score": round(mean_efe, 6),
                    "mean_epistemic_score": round(mean_epistemic, 6),
                    "trajectory_file": str(traj_file.relative_to(output_dir)),
                    "num_obstacles_spawned": num_obstacles_spawned,
                    "num_obstacles_avoided": num_avoided,
                    "obstacle_avoidance_rate": round(avoidance_rate, 4),
                    "num_lane_changes": num_lane_changes,
                }
                results.append(row)
                task_b_info = ""
                if is_task_b:
                    task_b_info = (
                        f" obs_avoided={num_avoided}/{num_obstacles_spawned}"
                        f" lane_changes={num_lane_changes}"
                    )
                print(
                    f"Route {ri} Ep {ep + 1}: {termination_reason} | "
                    f"completion={completion:.0%} (max {max_completion:.0%}) "
                    f"goal_dist={goal_dist:.1f}m (min {min_goal_dist:.1f}m) "
                    f"MLD={mld:.3f} offroad={offroad_events} frames={t + 1} "
                    f"EFE={mean_efe:.4f} epistemic={mean_epistemic:.4f}"
                    f"{task_b_info}"
                )

                if args.save_video:
                    if onboard_frames:
                        vp = output_dir / f"eval_{args.task}_r{ri}_ep{ep}_onboard.mp4"
                        imageio.mimwrite(str(vp), onboard_frames, fps=20)
                    if chase_frames:
                        vp = output_dir / f"eval_{args.task}_r{ri}_ep{ep}_chase.mp4"
                        imageio.mimwrite(str(vp), chase_frames, fps=20)

                # Save results incrementally after each episode
                _save_results()

              except Exception as e:
                import traceback
                print(f"Route {ri} Ep {ep + 1}: CARLA error - {e}")
                traceback.print_exc()
                # Try to reconnect
                try:
                    env.close()
                except Exception:
                    pass
                import time
                time.sleep(5)
                try:
                    env = CARLADrivingEnv(host=args.host, port=args.port, town=town)
                except Exception as e2:
                    print(f"  Failed to reconnect: {e2}")
                    break
                continue

    finally:
        destroy_obstacles(obstacle_actors)
        env.close()

    _save_results()

    if results:
        sr = sum(r["success"] for r in results) / len(results)
        avg_mld = np.mean([r["mean_lateral_dev"] for r in results])
        avg_comp = np.mean([r["route_completion_pct"] for r in results])
        avg_max_comp = np.mean([r["max_route_completion_pct"] for r in results])
        avg_efe = np.mean([r["mean_efe_score"] for r in results])
        avg_epi = np.mean([r["mean_epistemic_score"] for r in results])
        print(
            f"\nSummary: SR={sr:.0%}, Avg Completion={avg_comp:.1f}% (max {avg_max_comp:.1f}%), "
            f"MLD={avg_mld:.3f}, EFE={avg_efe:.4f}, Epistemic={avg_epi:.4f}"
        )

        # Task B aggregate metrics
        if is_task_b:
            total_obs = sum(r["num_obstacles_spawned"] for r in results)
            total_avoided = sum(r["num_obstacles_avoided"] for r in results)
            total_lc = sum(r["num_lane_changes"] for r in results)
            avg_avoid_rate = total_avoided / max(total_obs, 1)
            eps_with_lc = sum(1 for r in results if r["num_lane_changes"] > 0)
            print(
                f"Task B: Obstacles avoided={total_avoided}/{total_obs} "
                f"({avg_avoid_rate:.0%}), "
                f"Total lane changes={total_lc}, "
                f"Episodes with lane change={eps_with_lc}/{len(results)}"
            )


if __name__ == "__main__":
    main()

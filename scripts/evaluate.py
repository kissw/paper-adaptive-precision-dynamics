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
    from active_inference.evaluation.obstacles import spawn_obstacles, destroy_obstacles
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

                # Spawn obstacles for Task B
                if is_task_b:
                    obstacle_actors = spawn_obstacles(
                        env._world, env._vehicle, num_obstacles=args.num_obstacles
                    )
                    print(f"  Spawned {len(obstacle_actors)} obstacles for Task B")

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

                # Per-frame JSONL trajectory file
                traj_file = traj_dir / f"trajectory_{args.task}_r{ri}_ep{ep}.jsonl"
                traj_fh = open(traj_file, "w")

                try:
                    for t in range(args.max_frames):
                        img_tensor = torch.tensor(img, dtype=torch.float32)
                        state_tensor = torch.tensor(state, dtype=torch.float32)
                        plan_result = agent.step_with_info(img_tensor, state_tensor)
                        action = plan_result.action
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

                        collision_flag = info.get("collision", False)
                        lane_invasion_flag = info.get("lane_invasion", False)
                        if lane_invasion_flag:
                            offroad_events += 1
                        if collision_flag:
                            had_collision = True

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
                }
                results.append(row)
                print(
                    f"Route {ri} Ep {ep + 1}: {termination_reason} | "
                    f"completion={completion:.0%} (max {max_completion:.0%}) "
                    f"goal_dist={goal_dist:.1f}m (min {min_goal_dist:.1f}m) "
                    f"MLD={mld:.3f} offroad={offroad_events} frames={t + 1} "
                    f"EFE={mean_efe:.4f} epistemic={mean_epistemic:.4f}"
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
                print(f"Route {ri} Ep {ep + 1}: CARLA error - {e}")
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


if __name__ == "__main__":
    main()

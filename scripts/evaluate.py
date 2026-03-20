import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


TOWN_MAP = {"A": "Town04", "B": "Town03", "baseline": "Town06_Opt"}

STUCK_SPEED_THRESHOLD = 0.3
STUCK_FRAMES_LIMIT = 100
COLLISION_STUCK_FRAMES = 60
GOAL_DISTANCE_THRESHOLD = 10.0
MAX_FRAMES = 2000


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
        description="Evaluate Deep AIF agent on predefined CARLA routes"
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
    parser.add_argument("--output_dir", default="outputs/eval_v3")
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
    from active_inference.utils.transforms import denormalize_image

    town = TOWN_MAP[args.task]
    cfg = Config.from_yaml(args.config)
    agent = DeepAIFAgent(cfg)
    agent.load_checkpoint(args.checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eval_results.csv"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print(f"Validating routes for {town}...")
    route_validations = validate_routes(client, town)

    routes = EVAL_ROUTES.get(town, [])
    if args.route_index is not None:
        routes = [routes[args.route_index]]

    env = CARLADrivingEnv(host=args.host, port=args.port, town=town)
    results = []

    try:
        for ri, route in enumerate(routes):
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location
            route_wps = get_route_waypoints(client, town, route)

            for ep in range(args.episodes):
                obs = env.reset(spawn_point=start_spawn)
                env.set_goal(goal_loc)
                img, state = obs
                agent.reset()

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
                vehicle_loc = start_spawn.location

                for t in range(args.max_frames):
                    img_tensor = torch.tensor(img, dtype=torch.float32)
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    action = agent.step(img_tensor, state_tensor)
                    obs, info = env.step(action.cpu().numpy())
                    img, state = obs

                    vehicle_loc = env._vehicle.get_location()
                    goal_dist = _distance(vehicle_loc, goal_loc)

                    waypoint = env._world.get_map().get_waypoint(vehicle_loc)
                    if waypoint:
                        wp_loc = waypoint.transform.location
                        lateral_devs.append(
                            math.sqrt(
                                (vehicle_loc.x - wp_loc.x) ** 2 + (vehicle_loc.y - wp_loc.y) ** 2
                            )
                        )

                    if info.get("lane_invasion"):
                        offroad_events += 1

                    if info.get("collision"):
                        had_collision = True

                    speed = state[0]
                    if speed < STUCK_SPEED_THRESHOLD:
                        low_speed_counter += 1
                        if had_collision:
                            collision_stuck_counter += 1
                    else:
                        low_speed_counter = 0
                        collision_stuck_counter = 0

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

                completion = _route_completion(vehicle_loc, route_wps)
                mld = float(np.mean(lateral_devs)) if lateral_devs else 0.0
                success = termination_reason == "goal_reached"

                row = {
                    "task": args.task,
                    "town": town,
                    "episode": ep,
                    "route_id": ri,
                    "route_desc": route.get("desc", ""),
                    "success": int(success),
                    "route_completion_pct": round(completion * 100, 1),
                    "mean_lateral_dev": round(mld, 4),
                    "offroad_events": offroad_events,
                    "frames": t + 1,
                    "goal_distance": round(goal_dist, 2),
                    "termination": termination_reason,
                }
                results.append(row)
                print(
                    f"Route {ri} Ep {ep + 1}: {termination_reason} | "
                    f"completion={completion:.0%} goal_dist={goal_dist:.1f}m "
                    f"MLD={mld:.3f} offroad={offroad_events} frames={t + 1}"
                )

                if args.save_video:
                    if onboard_frames:
                        vp = output_dir / f"eval_{args.task}_r{ri}_ep{ep}_onboard.mp4"
                        imageio.mimwrite(str(vp), onboard_frames, fps=20)
                    if chase_frames:
                        vp = output_dir / f"eval_{args.task}_r{ri}_ep{ep}_chase.mp4"
                        imageio.mimwrite(str(vp), chase_frames, fps=20)

    finally:
        env.close()

    fieldnames = [
        "task",
        "town",
        "episode",
        "route_id",
        "route_desc",
        "success",
        "route_completion_pct",
        "mean_lateral_dev",
        "offroad_events",
        "frames",
        "goal_distance",
        "termination",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if results:
        sr = sum(r["success"] for r in results) / len(results)
        avg_mld = np.mean([r["mean_lateral_dev"] for r in results])
        avg_comp = np.mean([r["route_completion_pct"] for r in results])
        print(f"\nSummary: SR={sr:.0%}, Avg Completion={avg_comp:.1f}%, MLD={avg_mld:.3f}")


if __name__ == "__main__":
    main()

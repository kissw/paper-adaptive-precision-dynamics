import argparse
import sys
from pathlib import Path

import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Collect Task B (lane change) preference data by spawning static obstacles"
    )
    parser.add_argument("--town", default="Town06")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--output", default="data/preference_task_b.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=500)
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        import carla
        from agents.navigation.basic_agent import BasicAgent
    except ImportError:
        print("ERROR: carla package not installed.")
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

    ep_success_flags = []
    ep_task_labels = []

    obstacle_actors = []
    episode_idx = 0
    collected = 0

    try:
        while collected < args.num_samples:
            obs = env.reset()
            img, state = obs

            for a in obstacle_actors:
                try:
                    a.destroy()
                except Exception:
                    pass
            obstacle_actors.clear()

            # Spawn static obstacles in ego lane ahead to force lane changes
            ego_transform = env._vehicle.get_transform()
            ego_wp = env._world.get_map().get_waypoint(ego_transform.location)
            bp_lib = env._world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

            for i in range(args.num_obstacles):
                ahead_wp = ego_wp
                for _ in range(30 + i * 40):
                    nexts = ahead_wp.next(2.0)
                    if nexts:
                        ahead_wp = nexts[0]
                obstacle_transform = ahead_wp.transform
                obstacle_transform.location.z += 0.5
                try:
                    obstacle = env._world.try_spawn_actor(vehicle_bp, obstacle_transform)
                    if obstacle:
                        obstacle_actors.append(obstacle)
                except Exception:
                    pass

            agent = BasicAgent(env._vehicle, target_speed=30)
            spawn_points = env._world.get_map().get_spawn_points()
            dest = spawn_points[rng.integers(len(spawn_points))]
            agent.set_destination(dest.location)

            ep_start = collected
            ep_collisions = 0
            ep_lat_devs = []
            ep_lane_list = []

            for t in range(args.episode_len):
                if collected >= args.num_samples:
                    break

                control = agent.run_step()
                expert_action = carla_to_action(control.steer, control.throttle, control.brake)

                obs, info = env.step(expert_action)
                img, state = obs

                all_images.append(img)
                all_states.append(state)
                all_actions.append(expert_action)
                all_expert_actions.append(expert_action)
                all_episode_ids.append(episode_idx)

                if info.get("collision", False):
                    ep_collisions += 1

                waypoint = env._world.get_map().get_waypoint(env._vehicle.get_location())
                lane_id = waypoint.lane_id if waypoint else -1
                if waypoint:
                    loc = env._vehicle.get_location()
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
            is_success = ep_collisions == 0

            lane_changes = sum(
                1
                for i in range(1, len(ep_lane_list))
                if ep_lane_list[i] != ep_lane_list[i - 1]
                and ep_lane_list[i] != -1
                and ep_lane_list[i - 1] != -1
            )

            for _ in range(ep_frames):
                ep_success_flags.append(is_success)
                ep_task_labels.append(1)

            episode_idx += 1
            print(
                f"Ep {episode_idx:3d} | {ep_frames:3d}fr col={ep_collisions} "
                f"lat={mean_lat:.3f} lc={lane_changes} ok={is_success} | "
                f"{collected}/{args.num_samples}"
            )

    finally:
        for a in obstacle_actors:
            try:
                a.destroy()
            except Exception:
                pass
        env.close()

    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=np.stack(all_images), dtype=np.float32)
        f.create_dataset("states", data=np.stack(all_states), dtype=np.float32)
        f.create_dataset("actions", data=np.stack(all_actions), dtype=np.float32)
        f.create_dataset("expert_actions", data=np.stack(all_expert_actions), dtype=np.float32)
        f.create_dataset("episode_ids", data=np.array(all_episode_ids, dtype=np.int64))
        f.create_dataset("lateral_devs", data=np.array(all_lateral_devs, dtype=np.float32))
        f.create_dataset("lane_ids", data=np.array(all_lane_ids, dtype=np.int32))
        f.create_dataset("success_flags", data=np.array(ep_success_flags, dtype=bool))
        f.create_dataset("task_labels", data=np.array(ep_task_labels, dtype=np.int8))

        f.attrs["town"] = args.town
        f.attrs["total_frames"] = collected
        f.attrs["total_episodes"] = episode_idx
        f.attrs["purpose"] = "Task B preference data (lane change with obstacles)"

    print(f"\nSaved {collected} Task B frames ({episode_idx} episodes) to {args.output}")


if __name__ == "__main__":
    main()

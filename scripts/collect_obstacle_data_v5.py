"""Collect obstacle-avoidance data (v5) with camera-projection visibility labels.

Extends v4 by adding per-frame obstacle metadata including camera projection
visibility, bounding-box coordinates, and ego/obstacle world poses.

Key additions over v4
---------------------
- project_world_to_image()    : pure numpy, testable without CARLA
- compute_bbox_from_corners() : image-space bbox from 8 projected corners
- compute_visibility()        : five-condition visibility gate
- Per-frame HDF5 keys         : obstacle_visible, obstacle_distance, …
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import h5py

# ─────────────────────────────────────────────────────────────────────────────
# Camera projection utilities (pure numpy — importable without CARLA)
# ─────────────────────────────────────────────────────────────────────────────

# Constants matching CARLADrivingEnv (carla_env.py)
_CAM_FWD_OFFSET: float = 1.5   # metres forward from vehicle reference point
_CAM_UP_OFFSET:  float = 2.4   # metres above vehicle reference point
_CAM_FOV:        float = 90.0  # degrees


def project_world_to_image(
    point_world_xyz: tuple,
    ego_pos_xyz: tuple,
    ego_yaw_deg: float,
    image_size: int,
    fov_deg: float = _CAM_FOV,
    cam_fwd: float = _CAM_FWD_OFFSET,
    cam_up: float = _CAM_UP_OFFSET,
) -> tuple:
    """Project a 3D world point to image UV coordinates.

    Coordinate convention (matches carla_env.py fwd_proj / crosstrack_error):
      forward = (cos yaw, sin yaw, 0)   in world XY
      right   = (sin yaw, -cos yaw, 0)  in world XY
      Camera: X=right, Y=down, Z=forward.
    Camera is mounted cam_fwd metres ahead and cam_up metres above the vehicle
    reference point; no lateral offset and no rotation relative to body frame.

    Returns (u, v) in pixel coordinates, or (None, None) if the point is
    behind the camera (Z_cam <= 0).
    """
    W = H = image_size
    f = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    yaw = math.radians(ego_yaw_deg)
    dx = point_world_xyz[0] - ego_pos_xyz[0]
    dy = point_world_xyz[1] - ego_pos_xyz[1]
    dz = point_world_xyz[2] - ego_pos_xyz[2]

    fwd_body   = dx * math.cos(yaw) + dy * math.sin(yaw)
    right_body = dx * math.sin(yaw) - dy * math.cos(yaw)
    up_body    = dz

    Z_cam = fwd_body - cam_fwd
    X_cam = right_body
    Y_cam = -(up_body - cam_up)

    if Z_cam <= 0.0:
        return None, None

    u = f * X_cam / Z_cam + cx
    v = f * Y_cam / Z_cam + cy
    return u, v


def compute_bbox_from_corners(
    us: list,
    vs: list,
    image_size: int,
) -> tuple:
    """Compute image-space bounding box from projected 8-corner UV coordinates.

    Args:
        us: list of projected u values (None for behind-camera corners)
        vs: list of projected v values (None for behind-camera corners)
        image_size: W = H = image_size pixels

    Returns:
        (x1, y1, x2, y2) clipped to [0, image_size], or None if the
        resulting rectangle has zero area or lies entirely outside the image.
    """
    W = H = image_size
    valid = [(u, v) for u, v in zip(us, vs) if u is not None and v is not None]
    if not valid:
        return None

    all_u = [p[0] for p in valid]
    all_v = [p[1] for p in valid]

    x1, x2 = min(all_u), max(all_u)
    y1, y2 = min(all_v), max(all_v)

    if x2 < 0 or x1 >= W or y2 < 0 or y1 >= H:
        return None

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(W), x2)
    y2 = min(float(H), y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def compute_visibility(
    in_front: bool,
    distance: float,
    bbox: tuple,
    image_size: int,
    dist_threshold: float = 50.0,
    bbox_area_threshold: float = 20.0,
) -> bool:
    """Determine obstacle visibility given projection results.

    visible = (
        in_front == True                    [fwd_proj > 0]
        AND distance < dist_threshold
        AND bbox intersects image
        AND bbox_area > bbox_area_threshold
        AND y2 > 0.4 * image_size           [below crop_road boundary]
    )
    """
    if not in_front:
        return False
    if distance >= dist_threshold:
        return False
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    if area <= bbox_area_threshold:
        return False
    if y2 <= 0.4 * image_size:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CARLA-dependent helpers (only called inside main)
# ─────────────────────────────────────────────────────────────────────────────

def _get_bbox_corners_world(obstacle_actor) -> list:
    """Return 8 world-coordinate corners of a CARLA actor's bounding box."""
    import carla  # noqa: PLC0415
    bb = obstacle_actor.bounding_box
    ext = bb.extent
    loc = bb.location
    corners_local = [
        carla.Location(
            x=loc.x + sx * ext.x,
            y=loc.y + sy * ext.y,
            z=loc.z + sz * ext.z,
        )
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
    tf = obstacle_actor.get_transform()
    return [(c.x, c.y, c.z) for c in [tf.transform(cl) for cl in corners_local]]


def _empty_obstacle_meta(ego_x: float, ego_y: float, ego_yaw: float) -> dict:
    return {
        "obstacle_visible":  False,
        "obstacle_distance": float("inf"),
        "obstacle_in_front": False,
        "obstacle_lateral":  float("nan"),
        "nearest_obstacle_id": -1,
        "obstacle_bbox":     np.full(4, np.nan, dtype=np.float32),
        "obstacle_bbox_area": 0.0,
        "ego_x":             ego_x,
        "ego_y":             ego_y,
        "ego_yaw":           ego_yaw,
        "obstacle_x":        float("nan"),
        "obstacle_y":        float("nan"),
    }


def compute_nearest_obstacle_meta(
    obstacle_actors: list,
    ego_vehicle,
    image_size: int,
    fov_deg: float = _CAM_FOV,
    dist_threshold: float = 50.0,
    bbox_area_threshold: float = 20.0,
) -> dict:
    """Compute per-frame obstacle metadata for the nearest obstacle in scene."""
    loc = ego_vehicle.get_location()
    ego_pos = (loc.x, loc.y, loc.z)
    ego_yaw = float(ego_vehicle.get_transform().rotation.yaw)
    yaw_rad = math.radians(ego_yaw)

    if not obstacle_actors:
        return _empty_obstacle_meta(loc.x, loc.y, ego_yaw)

    nearest_dist = float("inf")
    nearest_idx = -1
    nearest_actor = None
    nearest_ox = float("nan")
    nearest_oy = float("nan")

    for oi, (actor, ox, oy) in enumerate(obstacle_actors):
        dx = ox - loc.x
        dy = oy - loc.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = oi
            nearest_actor = actor
            nearest_ox = ox
            nearest_oy = oy

    dx = nearest_ox - loc.x
    dy = nearest_oy - loc.y
    fwd_proj = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    lat_proj = dx * math.sin(yaw_rad) - dy * math.cos(yaw_rad)
    in_front = bool(fwd_proj > 0)

    corners_world = _get_bbox_corners_world(nearest_actor)
    us, vs = [], []
    for cxw, cyw, czw in corners_world:
        u, v = project_world_to_image(
            (cxw, cyw, czw), ego_pos, ego_yaw, image_size, fov_deg,
        )
        us.append(u)
        vs.append(v)

    bbox = compute_bbox_from_corners(us, vs, image_size)
    bbox_arr = (
        np.array(bbox, dtype=np.float32)
        if bbox is not None
        else np.full(4, np.nan, dtype=np.float32)
    )
    bbox_area = (
        float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if bbox is not None
        else 0.0
    )
    visible = compute_visibility(
        in_front=in_front,
        distance=nearest_dist,
        bbox=bbox,
        image_size=image_size,
        dist_threshold=dist_threshold,
        bbox_area_threshold=bbox_area_threshold,
    )

    return {
        "obstacle_visible":  visible,
        "obstacle_distance": float(nearest_dist),
        "obstacle_in_front": in_front,
        "obstacle_lateral":  float(lat_proj),
        "nearest_obstacle_id": nearest_idx,
        "obstacle_bbox":     bbox_arr,
        "obstacle_bbox_area": float(bbox_area),
        "ego_x":             float(loc.x),
        "ego_y":             float(loc.y),
        "ego_yaw":           ego_yaw,
        "obstacle_x":        float(nearest_ox),
        "obstacle_y":        float(nearest_oy),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main data collection
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect obstacle-avoidance data (v5) with visibility labels",
    )
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--output", default="data/expert_obstacle_v5.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=3000)
    parser.add_argument("--num_obstacles", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_speed", type=float, default=25.0)
    parser.add_argument("--visible_distance_threshold", type=float, default=50.0)
    parser.add_argument("--visible_bbox_area_threshold", type=float, default=20.0)
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

    # Per-episode accumulators
    all_images, all_states, all_actions, all_expert_actions = [], [], [], []
    all_episode_ids, all_lateral_devs, all_lane_ids = [], [], []
    all_noise_sigmas, ep_success_flags, ep_task_labels = [], [], []

    # v5 extra fields
    all_obs_visible, all_obs_distance, all_obs_in_front = [], [], []
    all_obs_lateral, all_nearest_obs_id, all_obs_bbox = [], [], []
    all_obs_bbox_area, all_ego_x, all_ego_y, all_ego_yaw = [], [], [], []
    all_obs_x, all_obs_y = [], []

    episode_idx = 0
    collected = 0
    obstacle_actors = []
    clean_episodes = 0
    total_attempts = 0

    FRACTION_SETS = [
        [0.30, 0.70], [0.25, 0.65], [0.35, 0.75], [0.20, 0.60], [0.40, 0.80],
    ]
    DETECT_DIST = 60.0

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
            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            ep_images, ep_states, ep_actions, ep_expert_actions = [], [], [], []
            ep_lat_devs, ep_lane_list = [], []
            ep_collisions = 0
            ep_meta: list[dict] = []

            current_obs_idx = 0
            passed_obstacles = set()
            redirect_done = set()
            total_attempts += 1

            for t in range(args.episode_len):
                loc = env._vehicle.get_location()
                yaw_rad = math.radians(env._vehicle.get_transform().rotation.yaw)

                # Track passed obstacles
                for oi, (ox, oy) in enumerate(obs_positions):
                    if oi in passed_obstacles:
                        continue
                    dx, dy = ox - loc.x, oy - loc.y
                    fwd_proj = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
                    if fwd_proj < -15.0:
                        passed_obstacles.add(oi)
                        if oi == current_obs_idx:
                            current_obs_idx += 1
                            agent.set_destination(goal_loc)

                # Redirect to adjacent lane when near obstacle
                if (current_obs_idx < n_spawned
                        and current_obs_idx not in passed_obstacles
                        and current_obs_idx not in redirect_done):
                    ox, oy = obs_positions[current_obs_idx]
                    dx, dy = ox - loc.x, oy - loc.y
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    fwd_proj = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
                    if fwd_proj > 0 and dist < DETECT_DIST:
                        obs_wp = carla_map.get_waypoint(carla.Location(x=ox, y=oy))
                        if obs_wp:
                            left = obs_wp.get_left_lane()
                            right = obs_wp.get_right_lane()
                            adj = None
                            if left and str(left.lane_type) == "Driving":
                                adj = left
                            elif right and str(right.lane_type) == "Driving":
                                adj = right
                            if adj:
                                ahead = adj.next(30.0)
                                if ahead:
                                    target = ahead[0].transform.location
                                    agent.set_destination(target)
                                    redirect_done.add(current_obs_idx)

                control = agent.run_step()
                expert_action = carla_to_action(
                    control.steer, control.throttle, control.brake,
                )
                noise = rng.normal(0, 0.02, size=2)
                noisy_action = np.clip(expert_action + noise, -1.0, 1.0)

                obs_meta = compute_nearest_obstacle_meta(
                    obstacle_actors=obstacle_actors,
                    ego_vehicle=env._vehicle,
                    image_size=args.image_size,
                    fov_deg=_CAM_FOV,
                    dist_threshold=args.visible_distance_threshold,
                    bbox_area_threshold=args.visible_bbox_area_threshold,
                )
                ep_meta.append(obs_meta)

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
                    ep_task_labels.append(1)
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
                clean_episodes += 1
                print(
                    f"  Ep {episode_idx:4d} [CLEAN] | {ep_frames:4d}fr "
                    f"lc={lane_changes} vis={sum(m['obstacle_visible'] for m in ep_meta):3d} "
                    f"| {collected}/{args.num_samples}"
                )
            else:
                reason = (
                    f"col={ep_collisions}" if ep_collisions > 0
                    else "no_lc" if lane_changes == 0
                    else f"short={ep_frames}"
                )
                print(f"  Attempt {total_attempts} [SKIP {reason}]")

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
        # v4 keys
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
        # v5 keys
        f.create_dataset("obstacle_visible", data=np.array(all_obs_visible[:n], dtype=bool))
        f.create_dataset("obstacle_distance", data=np.array(all_obs_distance[:n], dtype=np.float32))
        f.create_dataset("obstacle_in_front", data=np.array(all_obs_in_front[:n], dtype=bool))
        f.create_dataset("obstacle_lateral", data=np.array(all_obs_lateral[:n], dtype=np.float32))
        f.create_dataset("nearest_obstacle_id", data=np.array(all_nearest_obs_id[:n], dtype=np.int32))
        f.create_dataset("obstacle_bbox", data=np.stack(all_obs_bbox[:n]).astype(np.float32))
        f.create_dataset("obstacle_bbox_area", data=np.array(all_obs_bbox_area[:n], dtype=np.float32))
        f.create_dataset("ego_x", data=np.array(all_ego_x[:n], dtype=np.float32))
        f.create_dataset("ego_y", data=np.array(all_ego_y[:n], dtype=np.float32))
        f.create_dataset("ego_yaw", data=np.array(all_ego_yaw[:n], dtype=np.float32))
        f.create_dataset("obstacle_x", data=np.array(all_obs_x[:n], dtype=np.float32))
        f.create_dataset("obstacle_y", data=np.array(all_obs_y[:n], dtype=np.float32))

        f.attrs["town"] = args.town
        f.attrs["total_frames"] = n
        f.attrs["total_episodes"] = episode_idx
        f.attrs["clean_episodes"] = clean_episodes
        f.attrs["total_attempts"] = total_attempts
        f.attrs["collection_type"] = "waypoint_obstacle_avoidance_v5_visible_labeled"
        f.attrs["visible_distance_threshold"] = args.visible_distance_threshold
        f.attrs["visible_bbox_area_threshold"] = args.visible_bbox_area_threshold
        f.attrs["visible_bbox_y2_min"] = 0.4 * args.image_size
        f.attrs["camera_fov"] = _CAM_FOV
        f.attrs["camera_image_size"] = args.image_size
        f.attrs["camera_transform"] = f"x={_CAM_FWD_OFFSET} z={_CAM_UP_OFFSET} yaw=0"

    n_visible = int(np.sum(np.array(all_obs_visible[:n])))
    print(
        f"\nSaved {n} frames ({clean_episodes} episodes) to {args.output}"
        f"\n  Visible frames: {n_visible}/{n} ({100*n_visible/max(n,1):.1f}%)"
    )


if __name__ == "__main__":
    main()

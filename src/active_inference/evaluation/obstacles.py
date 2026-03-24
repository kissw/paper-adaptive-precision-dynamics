"""Reusable obstacle spawning for Task B evaluation.

Provides three spawning strategies:
- spawn_obstacles(): Original ego-relative placement (legacy)
- spawn_obstacles_multilane(): Only spawns where adjacent driving lane exists
- spawn_obstacles_on_route(): Deterministic route-fraction placement for evaluation
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _has_adjacent_driving_lane(waypoint) -> bool:
    """Check if waypoint has an adjacent driving lane for lane-change."""
    for get_lane in (waypoint.get_left_lane, waypoint.get_right_lane):
        adj = get_lane()
        if adj is not None and str(adj.lane_type) == "Driving":
            return True
    return False


def spawn_obstacles(
    world,
    vehicle,
    num_obstacles: int = 3,
    distances: list[int] | None = None,
) -> list:
    """Spawn static vehicles in ego lane at given distances ahead.

    Args:
        world: CARLA world object.
        vehicle: Ego vehicle actor.
        num_obstacles: Number of obstacles to spawn.
        distances: Waypoint-step distances from ego for each obstacle.
            Defaults to [30, 70, 110] (matching preference data collection).

    Returns:
        List of spawned CARLA actors. Caller is responsible for calling
        ``destroy_obstacles()`` when done.
    """
    if distances is None:
        distances = [30 + i * 40 for i in range(num_obstacles)]

    ego_wp = world.get_map().get_waypoint(vehicle.get_transform().location)
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    actors = []
    for dist in distances:
        ahead_wp = ego_wp
        for _ in range(dist):
            nexts = ahead_wp.next(2.0)
            if nexts:
                ahead_wp = nexts[0]
        obstacle_transform = ahead_wp.transform
        obstacle_transform.location.z += 0.5
        try:
            obstacle = world.try_spawn_actor(vehicle_bp, obstacle_transform)
            if obstacle:
                actors.append(obstacle)
        except Exception:
            pass
    return actors


def spawn_obstacles_multilane(
    world,
    vehicle,
    num_obstacles: int = 3,
    distances: list[int] | None = None,
) -> list:
    """Spawn static vehicles only where an adjacent driving lane exists.

    Uses wider spacing (50 waypoint steps) to give the agent room for
    lane-change maneuvers. Validates that each obstacle position has a
    neighbouring driving lane so the agent can physically avoid it.

    Args:
        world: CARLA world object.
        vehicle: Ego vehicle actor.
        num_obstacles: Number of obstacles to spawn.
        distances: Waypoint-step distances from ego for each obstacle.
            Defaults to [40, 90, 140] (wider spacing for lane-change room).

    Returns:
        List of spawned CARLA actors.
    """
    if distances is None:
        distances = [40 + i * 50 for i in range(num_obstacles)]

    ego_wp = world.get_map().get_waypoint(vehicle.get_transform().location)
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    actors = []
    for dist in distances:
        ahead_wp = ego_wp
        for _ in range(dist):
            nexts = ahead_wp.next(2.0)
            if nexts:
                ahead_wp = nexts[0]

        if not _has_adjacent_driving_lane(ahead_wp):
            logger.warning(
                "Skipping obstacle at step %d: no adjacent driving lane", dist
            )
            continue

        obstacle_transform = ahead_wp.transform
        obstacle_transform.location.z += 0.5
        try:
            obstacle = world.try_spawn_actor(vehicle_bp, obstacle_transform)
            if obstacle:
                actors.append(obstacle)
                logger.info(
                    "Spawned obstacle at step %d (lane %d, road %d)",
                    dist, ahead_wp.lane_id, ahead_wp.road_id,
                )
        except Exception:
            pass
    return actors


def spawn_obstacles_on_route(
    world,
    route_waypoints: list,
    num_obstacles: int = 3,
    fractions: list[float] | None = None,
) -> list:
    """Spawn obstacles at fixed fractions along a route for reproducibility.

    Places obstacles at deterministic route positions (default: 15%, 45%, 75%)
    and validates that each position has an adjacent driving lane.

    Args:
        world: CARLA world object.
        route_waypoints: List of (carla.Waypoint, RoadOption) from route planner.
        num_obstacles: Number of obstacles to spawn.
        fractions: Route fractions (0.0-1.0) for obstacle placement.
            Defaults to [0.15, 0.45, 0.75].

    Returns:
        List of spawned CARLA actors.
    """
    if fractions is None:
        fractions = [0.15, 0.45, 0.75]
    fractions = fractions[:num_obstacles]

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    n_wps = len(route_waypoints)

    actors = []
    for frac in fractions:
        idx = min(int(frac * n_wps), n_wps - 1)
        wp, _ = route_waypoints[idx]

        if not _has_adjacent_driving_lane(wp):
            # Search nearby waypoints for a valid multi-lane position
            found = False
            for offset in range(1, 20):
                for candidate_idx in (idx + offset, idx - offset):
                    if 0 <= candidate_idx < n_wps:
                        candidate_wp, _ = route_waypoints[candidate_idx]
                        if _has_adjacent_driving_lane(candidate_wp):
                            wp = candidate_wp
                            idx = candidate_idx
                            found = True
                            break
                if found:
                    break
            if not found:
                logger.warning(
                    "Skipping obstacle at fraction %.2f: no multi-lane position found",
                    frac,
                )
                continue

        obstacle_transform = wp.transform
        obstacle_transform.location.z += 0.5
        try:
            obstacle = world.try_spawn_actor(vehicle_bp, obstacle_transform)
            if obstacle:
                actors.append(obstacle)
                logger.info(
                    "Spawned obstacle at route fraction %.2f (idx %d/%d, lane %d)",
                    frac, idx, n_wps, wp.lane_id,
                )
        except Exception:
            pass

    return actors


def destroy_obstacles(obstacles: list) -> None:
    """Safely destroy all obstacle actors."""
    for actor in obstacles:
        try:
            actor.destroy()
        except Exception:
            pass
    obstacles.clear()

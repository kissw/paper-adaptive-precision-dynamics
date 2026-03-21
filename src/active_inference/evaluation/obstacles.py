"""Reusable obstacle spawning for Task B evaluation.

Extracted from scripts/collect_preference_data.py for reuse in
evaluation and data collection scripts.
"""

from __future__ import annotations


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


def destroy_obstacles(obstacles: list) -> None:
    """Safely destroy all obstacle actors."""
    for actor in obstacles:
        try:
            actor.destroy()
        except Exception:
            pass
    obstacles.clear()

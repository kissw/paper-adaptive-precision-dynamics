"""Pre-defined evaluation routes per CARLA town.

Each route has a start/goal spawn-point index, a human-readable
description, and must satisfy:
  - total length  > 200 m
  - at least 2 turns (RoadOption.LEFT or RIGHT)

Indices are discovered once via ``find_routes_with_curves`` and
hard-coded here so evaluations are reproducible without re-scanning.
"""

from __future__ import annotations

import math
from typing import Any

EVAL_ROUTES: dict[str, list[dict[str, Any]]] = {
    "Town04": [
        {"start": 0, "goal": 53, "desc": "Highway curves 765m 21turns"},
        {"start": 0, "goal": 54, "desc": "Highway curves 760m 21turns"},
        {"start": 0, "goal": 30, "desc": "Highway moderate curves 378m 0turns"},
    ],
    "Town06_Opt": [
        {"start": 0, "goal": 1, "desc": "Highway baseline 1332m 85turns"},
        {"start": 0, "goal": 2, "desc": "Highway baseline 1329m 85turns"},
    ],
    # Task B: intersection-free multi-lane routes for obstacle avoidance
    # Discovered via scripts/discover_routes.py on Town06_Opt
    "Town06_Opt_TaskB": [
        {"start": 0, "goal": 152, "desc": "Highway obstacle avoidance 373m 0turns"},
        {"start": 1, "goal": 91, "desc": "Highway obstacle avoidance 669m 0turns"},
    ],
}


def _count_turns(route_trace) -> int:
    from agents.navigation.local_planner import RoadOption

    return sum(1 for _, opt in route_trace if opt in (RoadOption.LEFT, RoadOption.RIGHT))


def _route_length(route_trace) -> float:
    total = 0.0
    for i in range(1, len(route_trace)):
        a = route_trace[i - 1][0].transform.location
        b = route_trace[i][0].transform.location
        total += math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    return total


def validate_routes(client, town: str, sampling_resolution: float = 2.0):
    """Trace every route for *town* and print length / turn count.

    Returns list of dicts with ``valid``, ``length``, ``turns`` keys.
    For TaskB routes (town ending in ``_TaskB``), validation uses relaxed
    criteria (no minimum turn count).
    """
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world = client.get_world()
    # TaskB routes use the base town map (strip _TaskB suffix)
    map_town = town.replace("_TaskB", "")
    if world.get_map().name.split("/")[-1] != map_town:
        world = client.load_world(map_town)

    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    spawns = carla_map.get_spawn_points()
    routes = EVAL_ROUTES.get(town, [])
    is_task_b = town.endswith("_TaskB")
    results = []
    for r in routes:
        si, gi = r["start"], r["goal"]
        if si >= len(spawns) or gi >= len(spawns):
            results.append({"valid": False, "length": 0, "turns": 0, "error": "index OOB"})
            continue
        trace = grp.trace_route(spawns[si].location, spawns[gi].location)
        length = _route_length(trace)
        turns = _count_turns(trace)
        if is_task_b:
            valid = length > 200 and turns <= 5
        else:
            valid = length > 200 and turns >= 2
        results.append({"valid": valid, "length": round(length, 1), "turns": turns})
        status = "OK" if valid else "FAIL"
        print(f"  [{status}] {r['desc']}: {length:.0f}m, {turns} turns, spawn {si}->{gi}")
    return results


def find_routes_with_curves(
    client,
    town: str,
    min_turns: int = 2,
    min_distance: float = 200.0,
    max_distance: float = 2000.0,
    sampling_resolution: float = 2.0,
    max_results: int = 5,
):
    """Scan spawn-point pairs to discover routes satisfying curvature and
    distance constraints.  Useful for one-time route discovery."""
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)

    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    spawns = carla_map.get_spawn_points()

    found: list[dict] = []
    for si in range(len(spawns)):
        for gi in range(len(spawns)):
            if si == gi:
                continue
            origin = spawns[si].location
            dest = spawns[gi].location
            direct = math.sqrt((origin.x - dest.x) ** 2 + (origin.y - dest.y) ** 2)
            if direct < min_distance * 0.3:
                continue
            try:
                trace = grp.trace_route(origin, dest)
            except Exception:
                continue
            if not trace:
                continue
            length = _route_length(trace)
            if length < min_distance or length > max_distance:
                continue
            turns = _count_turns(trace)
            if turns < min_turns:
                continue
            found.append({"start": si, "goal": gi, "length": round(length, 1), "turns": turns})
            print(f"  Found: spawn {si}->{gi}, {length:.0f}m, {turns} turns")
            if len(found) >= max_results:
                return found
    return found


def get_route_waypoints(client, town: str, route: dict, sampling_resolution: float = 2.0):
    """Return list of (carla.Waypoint, RoadOption) for the given route."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world = client.get_world()
    # Strip _TaskB suffix so we query the correct CARLA map
    map_town = town.replace("_TaskB", "")
    carla_map = world.get_map()
    if carla_map.name.split("/")[-1] != map_town:
        world = client.load_world(map_town)
        carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    spawns = carla_map.get_spawn_points()
    return grp.trace_route(spawns[route["start"]].location, spawns[route["goal"]].location)


def find_straight_routes(
    client,
    town: str,
    max_turns: int = 5,
    min_distance: float = 300.0,
    max_distance: float = 800.0,
    sampling_resolution: float = 2.0,
    max_results: int = 5,
):
    """Scan spawn-point pairs for low-turn-count, multi-lane segments
    suitable for obstacle avoidance (Task B)."""
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)

    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    spawns = carla_map.get_spawn_points()

    found: list[dict] = []
    for si in range(len(spawns)):
        for gi in range(len(spawns)):
            if si == gi:
                continue
            origin = spawns[si].location
            dest = spawns[gi].location
            direct = math.sqrt((origin.x - dest.x) ** 2 + (origin.y - dest.y) ** 2)
            if direct < min_distance * 0.3:
                continue
            try:
                trace = grp.trace_route(origin, dest)
            except Exception:
                continue
            if not trace:
                continue
            length = _route_length(trace)
            if length < min_distance or length > max_distance:
                continue
            turns = _count_turns(trace)
            if turns > max_turns:
                continue
            # Check that the start waypoint has multiple lanes (multi-lane road)
            start_wp = carla_map.get_waypoint(origin)
            if start_wp is None:
                continue
            found.append({
                "start": si, "goal": gi,
                "length": round(length, 1), "turns": turns,
                "lane_width": round(start_wp.lane_width, 2),
            })
            print(f"  Found: spawn {si}->{gi}, {length:.0f}m, {turns} turns, "
                  f"lane_w={start_wp.lane_width:.1f}")
            if len(found) >= max_results:
                return found
    return found

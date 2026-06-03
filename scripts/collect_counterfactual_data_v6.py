#!/usr/bin/env python3
"""Collect counterfactual action-sensitivity data (v6 smoke/main).

Purpose
-------
This script collects branch-structured counterfactual driving sequences:

    sequence = BasicAgent context frames + counterfactual branch frames

For each anchor state, the script restores the context-start CARLA actor state,
replays the same context actions, and then rolls out steering branches sampled
from continuous steering bins. This avoids branch-to-branch hidden vehicle
physics artifacts that can occur when directly teleporting to the anchor.

Default smoke setting:
    clean anchors    = 5
    obstacle anchors = 5
    steer bins       = 8
    samples/bin      = 3
    branches/anchor  = 24
    context_len      = 40
    branch_horizon   = 10
    diagnostic_horizon = 20

Stored training frames:
    (5 + 5) anchors * 24 branches * (40 + 10) frames = 12,000 frames

Important convention
--------------------
actions[t] is the action that was applied by env.step(actions[t]) and produced
images[t], states[t]. This matches the fixed world-model training code.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _clear_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def clear_sensor_queues(env) -> None:
    """Clear camera queues so restored branches do not read stale frames."""
    if hasattr(env, "_image_queue"):
        _clear_queue(env._image_queue)
    if hasattr(env, "_chase_image_queue"):
        _clear_queue(env._chase_image_queue)


def default_obstacle_meta(env, image_size: int) -> dict[str, Any]:
    """Return v5-compatible no-obstacle metadata with current ego pose."""
    loc = env._vehicle.get_location()
    yaw = float(env._vehicle.get_transform().rotation.yaw)
    return {
        "obstacle_visible": False,
        "obstacle_distance": float("inf"),
        "obstacle_in_front": False,
        "obstacle_lateral": float("nan"),
        "nearest_obstacle_id": -1,
        "visible_obstacle_id": -1,
        "obstacle_bbox": np.full(4, np.nan, dtype=np.float32),
        "obstacle_bbox_area": 0.0,
        "ego_x": float(loc.x),
        "ego_y": float(loc.y),
        "ego_yaw": yaw,
        "obstacle_x": float("nan"),
        "obstacle_y": float("nan"),
    }


def get_lane_info(env, carla_map) -> tuple[int, float]:
    """Return (lane_id, lateral deviation to nearest waypoint center)."""
    loc = env._vehicle.get_location()
    waypoint = carla_map.get_waypoint(loc)
    if waypoint is None:
        return -1, 0.0

    wp_loc = waypoint.transform.location
    lane_id = int(waypoint.lane_id)
    lat_dev = math.sqrt((loc.x - wp_loc.x) ** 2 + (loc.y - wp_loc.y) ** 2)
    return lane_id, float(lat_dev)


def goal_distance(vehicle, goal_loc) -> float:
    loc = vehicle.get_location()
    return math.sqrt((loc.x - goal_loc.x) ** 2 + (loc.y - goal_loc.y) ** 2)


def sample_steers(
    rng: np.random.Generator,
    steer_min: float,
    steer_max: float,
    steer_bins: int,
    samples_per_bin: int,
) -> list[tuple[int, float]]:
    """Sample rounded continuous steering values from equal-width bins."""
    edges = np.linspace(steer_min, steer_max, steer_bins + 1)
    results: list[tuple[int, float]] = []

    for bin_id in range(steer_bins):
        low = float(edges[bin_id])
        high = float(edges[bin_id + 1])
        used: set[float] = set()
        for _ in range(samples_per_bin):
            for _attempt in range(100):
                steer = round(float(rng.uniform(low, high)), 3)
                if steer not in used:
                    used.add(steer)
                    break
            results.append((bin_id, steer))
    return results


def route_fraction_sets(num_obstacles: int) -> list[list[float]]:
    base = [
        [0.30, 0.70],
        [0.25, 0.65],
        [0.35, 0.75],
        [0.20, 0.60],
        [0.40, 0.80],
    ]
    if num_obstacles <= 2:
        return base
    return [
        np.linspace(0.20, 0.80, num_obstacles).tolist(),
        np.linspace(0.15, 0.75, num_obstacles).tolist(),
        np.linspace(0.25, 0.85, num_obstacles).tolist(),
    ]


@dataclass
class ActorState:
    transform: Any
    velocity: Any
    angular_velocity: Any
    control: Any | None


@dataclass
class WorldSnapshot:
    """CARLA actor states required for deterministic-ish context replay."""

    ego: ActorState
    obstacles: list[tuple[Any, ActorState]]


@dataclass
class AnchorSnapshot:
    ego: ActorState
    obstacles: list[tuple[Any, ActorState]]
    anchor_image: np.ndarray
    anchor_state: np.ndarray
    anchor_expert_action: np.ndarray
    anchor_nominal_action: np.ndarray
    anchor_meta: dict[str, Any]
    anchor_lane_id: int
    anchor_lateral_dev: float


def _actor_state(actor) -> ActorState:
    control = None
    try:
        control = actor.get_control()
    except Exception:
        control = None
    return ActorState(
        transform=actor.get_transform(),
        velocity=actor.get_velocity(),
        angular_velocity=actor.get_angular_velocity(),
        control=control,
    )


def make_world_snapshot(env, obstacle_actors: list) -> WorldSnapshot:
    """Snapshot ego + obstacle actor states before a control is applied."""
    obstacles = []
    for item in obstacle_actors:
        actor = item[0] if isinstance(item, tuple) else item
        obstacles.append((actor, _actor_state(actor)))
    return WorldSnapshot(ego=_actor_state(env._vehicle), obstacles=obstacles)


def restore_world_snapshot(env, snapshot: WorldSnapshot) -> None:
    """Restore ego + obstacle actor states and clear transient event/sensor queues."""
    import carla

    for actor, st in snapshot.obstacles:
        if actor is None or not actor.is_alive:
            continue
        actor.set_transform(st.transform)
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    ego = env._vehicle
    ego.set_transform(snapshot.ego.transform)
    ego.set_target_velocity(snapshot.ego.velocity)
    ego.set_target_angular_velocity(snapshot.ego.angular_velocity)
    if snapshot.ego.control is not None:
        ego.apply_control(snapshot.ego.control)

    env._collision_flag = False
    env._lane_invasion_flag = False
    clear_sensor_queues(env)


def make_snapshot(
    env,
    obstacle_actors: list,
    anchor_image: np.ndarray,
    anchor_state: np.ndarray,
    anchor_expert_action: np.ndarray,
    anchor_nominal_action: np.ndarray,
    anchor_meta: dict[str, Any],
    anchor_lane_id: int,
    anchor_lateral_dev: float,
) -> AnchorSnapshot:
    obstacles = []
    for item in obstacle_actors:
        actor = item[0] if isinstance(item, tuple) else item
        obstacles.append((actor, _actor_state(actor)))

    return AnchorSnapshot(
        ego=_actor_state(env._vehicle),
        obstacles=obstacles,
        anchor_image=anchor_image.copy(),
        anchor_state=anchor_state.copy(),
        anchor_expert_action=anchor_expert_action.copy(),
        anchor_nominal_action=anchor_nominal_action.copy(),
        anchor_meta=dict(anchor_meta),
        anchor_lane_id=int(anchor_lane_id),
        anchor_lateral_dev=float(anchor_lateral_dev),
    )


def restore_snapshot(env, snapshot: AnchorSnapshot) -> None:
    """Restore ego and obstacle actors to the exact anchor state."""
    import carla

    for actor, st in snapshot.obstacles:
        if actor is None or not actor.is_alive:
            continue
        actor.set_transform(st.transform)
        actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass

    ego = env._vehicle
    ego.set_transform(snapshot.ego.transform)
    ego.set_target_velocity(snapshot.ego.velocity)
    ego.set_target_angular_velocity(snapshot.ego.angular_velocity)
    if snapshot.ego.control is not None:
        ego.apply_control(snapshot.ego.control)

    env._collision_flag = False
    env._lane_invasion_flag = False
    clear_sensor_queues(env)


def make_frame(
    image: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    expert_action: np.ndarray,
    lane_id: int,
    lateral_dev: float,
    meta: dict[str, Any],
    step_collision: bool,
    step_lane_invasion: bool,
) -> dict[str, Any]:
    return {
        "image": image.astype(np.float32),
        "state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "expert_action": expert_action.astype(np.float32),
        "lane_id": int(lane_id),
        "lateral_dev": float(lateral_dev),
        "meta": dict(meta),
        "step_collision": bool(step_collision),
        "step_lane_invasion": bool(step_lane_invasion),
    }


class H5Accumulator:
    """Append rows and write v5-compatible + counterfactual HDF5 keys."""

    def __init__(self) -> None:
        self.data: dict[str, list] = {k: [] for k in [
            "images", "states", "actions", "expert_actions", "episode_ids",
            "lateral_devs", "lane_ids", "noise_sigmas", "success_flags",
            "task_labels", "target_speeds", "obstacle_visible",
            "obstacle_distance", "obstacle_in_front", "obstacle_lateral",
            "nearest_obstacle_id", "visible_obstacle_id", "obstacle_bbox",
            "obstacle_bbox_area", "ego_x", "ego_y", "ego_yaw", "obstacle_x",
            "obstacle_y", "is_counterfactual", "cf_is_context", "cf_is_branch",
            "cf_anchor_id", "cf_branch_id", "cf_branch_step", "cf_bin_id",
            "cf_sampled_steer", "cf_nominal_steer", "cf_nominal_accel",
            "cf_anchor_speed", "cf_anchor_obstacle_visible",
            "cf_anchor_obstacle_distance", "cf_branch_collision_10",
            "cf_branch_lane_invasion_10", "cf_branch_collision_diag",
            "cf_branch_lane_invasion_diag", "cf_final_abs_cte_delta_10",
            "cf_final_abs_cte_delta_diag", "cf_max_abs_cte_delta_10",
            "cf_max_abs_cte_delta_diag", "cf_step_collision",
            "cf_step_lane_invasion", "cf_source_type",
        ]}

    def append_sequence(
        self,
        *,
        episode_id: int,
        anchor_id: int,
        branch_id: int,
        bin_id: int,
        sampled_steer: float,
        context_frames: list[dict[str, Any]],
        branch_frames: list[dict[str, Any]],
        task_label: int,
        target_speed: float,
        nominal_noise_sigma: tuple[float, float],
        branch_diag: dict[str, Any],
        anchor_snapshot: AnchorSnapshot,
    ) -> None:
        frames = context_frames + branch_frames
        context_len = len(context_frames)
        branch_success = not (
            branch_diag["collision_10"] or branch_diag["lane_invasion_10"]
        )
        anchor_speed = float(anchor_snapshot.anchor_state[0])
        anchor_obs_visible = bool(anchor_snapshot.anchor_meta["obstacle_visible"])
        anchor_obs_dist = float(anchor_snapshot.anchor_meta["obstacle_distance"])
        nominal_steer = float(anchor_snapshot.anchor_expert_action[0])
        nominal_accel = float(anchor_snapshot.anchor_expert_action[1])

        for i, fr in enumerate(frames):
            is_context = i < context_len
            branch_step = i - context_len + 1
            meta = fr["meta"]
            d = self.data

            d["images"].append(fr["image"])
            d["states"].append(fr["state"])
            d["actions"].append(fr["action"])
            d["expert_actions"].append(fr["expert_action"])
            d["episode_ids"].append(int(episode_id))
            d["lateral_devs"].append(float(fr["lateral_dev"]))
            d["lane_ids"].append(int(fr["lane_id"]))
            d["noise_sigmas"].append([float(nominal_noise_sigma[0]), float(nominal_noise_sigma[1])])
            d["success_flags"].append(bool(branch_success))
            d["task_labels"].append(int(task_label))
            d["target_speeds"].append(float(target_speed))

            d["obstacle_visible"].append(bool(meta["obstacle_visible"]))
            d["obstacle_distance"].append(float(meta["obstacle_distance"]))
            d["obstacle_in_front"].append(bool(meta["obstacle_in_front"]))
            d["obstacle_lateral"].append(float(meta["obstacle_lateral"]))
            d["nearest_obstacle_id"].append(int(meta["nearest_obstacle_id"]))
            d["visible_obstacle_id"].append(int(meta["visible_obstacle_id"]))
            d["obstacle_bbox"].append(np.asarray(meta["obstacle_bbox"], dtype=np.float32))
            d["obstacle_bbox_area"].append(float(meta["obstacle_bbox_area"]))
            d["ego_x"].append(float(meta["ego_x"]))
            d["ego_y"].append(float(meta["ego_y"]))
            d["ego_yaw"].append(float(meta["ego_yaw"]))
            d["obstacle_x"].append(float(meta["obstacle_x"]))
            d["obstacle_y"].append(float(meta["obstacle_y"]))

            d["is_counterfactual"].append(True)
            d["cf_is_context"].append(bool(is_context))
            d["cf_is_branch"].append(not is_context)
            d["cf_anchor_id"].append(int(anchor_id))
            d["cf_branch_id"].append(int(branch_id))
            d["cf_branch_step"].append(int(branch_step))
            d["cf_bin_id"].append(int(bin_id))
            d["cf_sampled_steer"].append(float(sampled_steer))
            d["cf_nominal_steer"].append(nominal_steer)
            d["cf_nominal_accel"].append(nominal_accel)
            d["cf_anchor_speed"].append(anchor_speed)
            d["cf_anchor_obstacle_visible"].append(anchor_obs_visible)
            d["cf_anchor_obstacle_distance"].append(anchor_obs_dist)
            d["cf_branch_collision_10"].append(bool(branch_diag["collision_10"]))
            d["cf_branch_lane_invasion_10"].append(bool(branch_diag["lane_invasion_10"]))
            d["cf_branch_collision_diag"].append(bool(branch_diag["collision_diag"]))
            d["cf_branch_lane_invasion_diag"].append(bool(branch_diag["lane_invasion_diag"]))
            d["cf_final_abs_cte_delta_10"].append(float(branch_diag["final_abs_cte_delta_10"]))
            d["cf_final_abs_cte_delta_diag"].append(float(branch_diag["final_abs_cte_delta_diag"]))
            d["cf_max_abs_cte_delta_10"].append(float(branch_diag["max_abs_cte_delta_10"]))
            d["cf_max_abs_cte_delta_diag"].append(float(branch_diag["max_abs_cte_delta_diag"]))
            d["cf_step_collision"].append(bool(fr["step_collision"]))
            d["cf_step_lane_invasion"].append(bool(fr["step_lane_invasion"]))
            d["cf_source_type"].append(int(task_label))

    def write(self, path: str | Path, attrs: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.data
        if not d["images"]:
            raise RuntimeError("No frames collected; refusing to write empty HDF5.")

        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=np.stack(d["images"]), dtype=np.float32)
            f.create_dataset("states", data=np.stack(d["states"]), dtype=np.float32)
            f.create_dataset("actions", data=np.stack(d["actions"]), dtype=np.float32)
            f.create_dataset("expert_actions", data=np.stack(d["expert_actions"]), dtype=np.float32)
            f.create_dataset("episode_ids", data=np.asarray(d["episode_ids"], dtype=np.int64))
            f.create_dataset("lateral_devs", data=np.asarray(d["lateral_devs"], dtype=np.float32))
            f.create_dataset("lane_ids", data=np.asarray(d["lane_ids"], dtype=np.int32))
            f.create_dataset("noise_sigmas", data=np.asarray(d["noise_sigmas"], dtype=np.float32))
            f.create_dataset("success_flags", data=np.asarray(d["success_flags"], dtype=bool))
            f.create_dataset("task_labels", data=np.asarray(d["task_labels"], dtype=np.int8))
            f.create_dataset("target_speeds", data=np.asarray(d["target_speeds"], dtype=np.float32))
            f.create_dataset("obstacle_visible", data=np.asarray(d["obstacle_visible"], dtype=bool))
            f.create_dataset("obstacle_distance", data=np.asarray(d["obstacle_distance"], dtype=np.float32))
            f.create_dataset("obstacle_in_front", data=np.asarray(d["obstacle_in_front"], dtype=bool))
            f.create_dataset("obstacle_lateral", data=np.asarray(d["obstacle_lateral"], dtype=np.float32))
            f.create_dataset("nearest_obstacle_id", data=np.asarray(d["nearest_obstacle_id"], dtype=np.int32))
            f.create_dataset("visible_obstacle_id", data=np.asarray(d["visible_obstacle_id"], dtype=np.int32))
            f.create_dataset("obstacle_bbox", data=np.stack(d["obstacle_bbox"]).astype(np.float32))
            f.create_dataset("obstacle_bbox_area", data=np.asarray(d["obstacle_bbox_area"], dtype=np.float32))
            f.create_dataset("ego_x", data=np.asarray(d["ego_x"], dtype=np.float32))
            f.create_dataset("ego_y", data=np.asarray(d["ego_y"], dtype=np.float32))
            f.create_dataset("ego_yaw", data=np.asarray(d["ego_yaw"], dtype=np.float32))
            f.create_dataset("obstacle_x", data=np.asarray(d["obstacle_x"], dtype=np.float32))
            f.create_dataset("obstacle_y", data=np.asarray(d["obstacle_y"], dtype=np.float32))

            for key in [
                "is_counterfactual", "cf_is_context", "cf_is_branch", "cf_anchor_id",
                "cf_branch_id", "cf_branch_step", "cf_bin_id", "cf_sampled_steer",
                "cf_nominal_steer", "cf_nominal_accel", "cf_anchor_speed",
                "cf_anchor_obstacle_visible", "cf_anchor_obstacle_distance",
                "cf_branch_collision_10", "cf_branch_lane_invasion_10",
                "cf_branch_collision_diag", "cf_branch_lane_invasion_diag",
                "cf_final_abs_cte_delta_10", "cf_final_abs_cte_delta_diag",
                "cf_max_abs_cte_delta_10", "cf_max_abs_cte_delta_diag",
                "cf_step_collision", "cf_step_lane_invasion", "cf_source_type",
            ]:
                values = d[key]
                if key.startswith("cf_is") or key.startswith("cf_branch_collision") or key.startswith("cf_branch_lane") or key.startswith("cf_step") or key in ["is_counterfactual", "cf_anchor_obstacle_visible"]:
                    arr = np.asarray(values, dtype=bool)
                elif key in ["cf_anchor_id", "cf_branch_id"]:
                    arr = np.asarray(values, dtype=np.int64)
                elif key in ["cf_branch_step", "cf_bin_id", "cf_source_type"]:
                    arr = np.asarray(values, dtype=np.int32 if key != "cf_source_type" else np.int8)
                else:
                    arr = np.asarray(values, dtype=np.float32)
                f.create_dataset(key, data=arr)

            for key, value in attrs.items():
                f.attrs[key] = value


def compute_meta_for_mode(
    *,
    mode: str,
    env,
    obstacle_actors: list,
    image_size: int,
    visible_distance_threshold: float,
    visible_bbox_area_threshold: float,
) -> dict[str, Any]:
    if mode == "obstacle":
        from collect_obstacle_data_v5 import _CAM_FOV, compute_obstacle_meta

        return compute_obstacle_meta(
            obstacle_actors=obstacle_actors,
            ego_vehicle=env._vehicle,
            image_size=image_size,
            fov_deg=_CAM_FOV,
            dist_threshold=visible_distance_threshold,
            bbox_area_threshold=visible_bbox_area_threshold,
        )
    return default_obstacle_meta(env, image_size)


def collect_branch_with_context_replay(
    *,
    env,
    carla_map,
    context_start_snapshot: WorldSnapshot,
    context_frames_nominal: list[dict[str, Any]],
    cf_action: np.ndarray,
    mode: str,
    obstacle_actors: list,
    image_size: int,
    branch_horizon: int,
    diagnostic_horizon: int,
    visible_distance_threshold: float,
    visible_bbox_area_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], AnchorSnapshot | None]:
    """Replay context from a pre-context snapshot, then collect one CF branch.

    This is the critical fix over the old anchor-restore method.

    Old behavior:
        restore anchor snapshot -> apply cf_action

    New behavior:
        restore context-start snapshot -> replay the same context actions -> apply cf_action

    Replaying the context lets CARLA rebuild wheel/vehicle internal dynamics before
    the counterfactual branch, which makes branch responses more consistent.
    """
    restore_world_snapshot(env, context_start_snapshot)

    replayed_context_frames: list[dict[str, Any]] = []

    # Recreate the context by replaying the exact actions that produced the
    # nominal context frames. These replayed frames are what get saved, not the
    # copied nominal frames, so each sequence is physically continuous.
    for nominal_frame in context_frames_nominal:
        context_action = np.asarray(nominal_frame["action"], dtype=np.float32)
        expert_action = np.asarray(nominal_frame["expert_action"], dtype=np.float32)

        obs, info = env.step(context_action)
        img, state = obs
        meta = compute_meta_for_mode(
            mode=mode,
            env=env,
            obstacle_actors=obstacle_actors,
            image_size=image_size,
            visible_distance_threshold=visible_distance_threshold,
            visible_bbox_area_threshold=visible_bbox_area_threshold,
        )
        lane_id, lat_dev = get_lane_info(env, carla_map)
        step_collision = bool(info.get("collision", False))
        step_lane_invasion = bool(info.get("lane_invasion", False))

        replayed_context_frames.append(
            make_frame(
                image=img,
                state=state,
                action=context_action,
                expert_action=expert_action,
                lane_id=lane_id,
                lateral_dev=lat_dev,
                meta=meta,
                step_collision=step_collision,
                step_lane_invasion=step_lane_invasion,
            )
        )

        # If the replayed nominal context already collides, this anchor is not
        # reliable for counterfactual comparison. Drop this branch.
        if step_collision:
            diag = {
                "collision_10": True,
                "lane_invasion_10": step_lane_invasion,
                "collision_diag": True,
                "lane_invasion_diag": step_lane_invasion,
                "final_abs_cte_delta_10": float("nan"),
                "final_abs_cte_delta_diag": float("nan"),
                "max_abs_cte_delta_10": float("nan"),
                "max_abs_cte_delta_diag": float("nan"),
            }
            return replayed_context_frames, [], diag, None

    if len(replayed_context_frames) == 0:
        diag = {
            "collision_10": True,
            "lane_invasion_10": True,
            "collision_diag": True,
            "lane_invasion_diag": True,
            "final_abs_cte_delta_10": float("nan"),
            "final_abs_cte_delta_diag": float("nan"),
            "max_abs_cte_delta_10": float("nan"),
            "max_abs_cte_delta_diag": float("nan"),
        }
        return replayed_context_frames, [], diag, None

    anchor_frame = replayed_context_frames[-1]
    branch_anchor_snapshot = make_snapshot(
        env=env,
        obstacle_actors=obstacle_actors,
        anchor_image=anchor_frame["image"],
        anchor_state=anchor_frame["state"],
        anchor_expert_action=anchor_frame["expert_action"],
        anchor_nominal_action=anchor_frame["action"],
        anchor_meta=anchor_frame["meta"],
        anchor_lane_id=anchor_frame["lane_id"],
        anchor_lateral_dev=anchor_frame["lateral_dev"],
    )

    branch_frames: list[dict[str, Any]] = []
    cte0 = float(anchor_frame["state"][3])
    cte_deltas_10: list[float] = []
    cte_deltas_diag: list[float] = []
    collision_10 = False
    lane_invasion_10 = False
    collision_diag = False
    lane_invasion_diag = False

    for h in range(diagnostic_horizon):
        obs, info = env.step(cf_action)
        img, state = obs
        meta = compute_meta_for_mode(
            mode=mode,
            env=env,
            obstacle_actors=obstacle_actors,
            image_size=image_size,
            visible_distance_threshold=visible_distance_threshold,
            visible_bbox_area_threshold=visible_bbox_area_threshold,
        )
        lane_id, lat_dev = get_lane_info(env, carla_map)
        step_collision = bool(info.get("collision", False))
        step_lane_invasion = bool(info.get("lane_invasion", False))
        abs_cte_delta = abs(float(state[3]) - cte0)

        if h < branch_horizon:
            cte_deltas_10.append(abs_cte_delta)
            collision_10 = collision_10 or step_collision
            lane_invasion_10 = lane_invasion_10 or step_lane_invasion
            branch_frames.append(
                make_frame(
                    image=img,
                    state=state,
                    action=cf_action,
                    expert_action=branch_anchor_snapshot.anchor_expert_action,
                    lane_id=lane_id,
                    lateral_dev=lat_dev,
                    meta=meta,
                    step_collision=step_collision,
                    step_lane_invasion=step_lane_invasion,
                )
            )

        cte_deltas_diag.append(abs_cte_delta)
        collision_diag = collision_diag or step_collision
        lane_invasion_diag = lane_invasion_diag or step_lane_invasion

    diag = {
        "collision_10": collision_10,
        "lane_invasion_10": lane_invasion_10,
        "collision_diag": collision_diag,
        "lane_invasion_diag": lane_invasion_diag,
        "final_abs_cte_delta_10": cte_deltas_10[-1] if cte_deltas_10 else float("nan"),
        "final_abs_cte_delta_diag": cte_deltas_diag[-1] if cte_deltas_diag else float("nan"),
        "max_abs_cte_delta_10": max(cte_deltas_10) if cte_deltas_10 else float("nan"),
        "max_abs_cte_delta_diag": max(cte_deltas_diag) if cte_deltas_diag else float("nan"),
    }
    return replayed_context_frames, branch_frames, diag, branch_anchor_snapshot

def should_take_anchor(
    *,
    mode: str,
    context_len: int,
    context_buffer: deque,
    stable_ticks: int,
    stable_speed_ticks: int,
    tick_idx: int,
    last_anchor_tick: int,
    anchor_stride: int,
    meta: dict[str, Any],
    min_obstacle_distance: float,
    max_obstacle_distance: float,
) -> bool:
    base_ok = (
        len(context_buffer) >= context_len
        and stable_ticks >= stable_speed_ticks
        and tick_idx - last_anchor_tick >= anchor_stride
    )
    if not base_ok:
        return False
    if mode == "clean":
        return True
    if not bool(meta["obstacle_visible"]):
        return False
    if not bool(meta["obstacle_in_front"]):
        return False
    dist = float(meta["obstacle_distance"])
    return min_obstacle_distance <= dist <= max_obstacle_distance


def summarize_diag(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{mode}_branches": 0,
            f"{mode}_mean_max_abs_cte_delta_10": float("nan"),
            f"{mode}_mean_max_abs_cte_delta_diag": float("nan"),
            f"{mode}_lane_invasion_rate_10": float("nan"),
            f"{mode}_collision_rate_10": float("nan"),
        }
    return {
        f"{mode}_branches": len(rows),
        f"{mode}_mean_max_abs_cte_delta_10": float(np.mean([r["max_abs_cte_delta_10"] for r in rows])),
        f"{mode}_mean_max_abs_cte_delta_diag": float(np.mean([r["max_abs_cte_delta_diag"] for r in rows])),
        f"{mode}_lane_invasion_rate_10": float(np.mean([float(r["lane_invasion_10"]) for r in rows])),
        f"{mode}_collision_rate_10": float(np.mean([float(r["collision_10"]) for r in rows])),
        f"{mode}_lane_invasion_rate_diag": float(np.mean([float(r["lane_invasion_diag"]) for r in rows])),
        f"{mode}_collision_rate_diag": float(np.mean([float(r["collision_diag"]) for r in rows])),
    }


def collect_mode(
    *,
    args,
    mode: str,
    target_anchors: int,
    env,
    carla_map,
    client,
    rng: np.random.Generator,
    accum: H5Accumulator,
    global_anchor_start: int,
    global_branch_start: int,
    global_episode_start: int,
) -> tuple[int, int, int, dict[str, Any]]:
    import carla
    from agents.navigation.basic_agent import BasicAgent

    from active_inference.data.carla_env import carla_to_action
    from active_inference.evaluation.obstacles import destroy_obstacles, spawn_obstacles_on_route
    from active_inference.evaluation.routes import EVAL_ROUTES, get_route_waypoints

    routes = EVAL_ROUTES.get(args.route_key, [])
    if not routes:
        raise ValueError(f"No routes found for route_key={args.route_key}")

    anchors_done = 0
    attempts = 0
    global_anchor_id = global_anchor_start
    global_branch_id = global_branch_start
    global_episode_id = global_episode_start
    diag_rows: list[dict[str, Any]] = []
    fraction_sets = route_fraction_sets(args.num_obstacles)
    obstacle_actors: list = []

    print("=" * 100)
    print(f"Collect mode={mode} | target_anchors={target_anchors}")

    try:
        while anchors_done < target_anchors:
            attempts += 1
            route_idx = int(rng.integers(len(routes)))
            route = routes[route_idx]
            spawns = env._world.get_map().get_spawn_points()
            start_spawn = spawns[route["start"]]
            goal_loc = spawns[route["goal"]].location

            destroy_obstacles(obstacle_actors)
            env.reset(spawn_point=start_spawn)
            env.set_goal(goal_loc)
            route_wps = get_route_waypoints(client, args.town, route)

            obs_positions: list[tuple[float, float]] = []
            if mode == "obstacle":
                fracs = fraction_sets[attempts % len(fraction_sets)]
                obstacle_actors = spawn_obstacles_on_route(
                    env._world,
                    route_wps,
                    num_obstacles=args.num_obstacles,
                    fractions=fracs,
                )
                if len(obstacle_actors) == 0:
                    print(f"  Attempt {attempts} [SKIP no obstacles spawned]")
                    continue
                obs_positions = [(ix, iy) for _, ix, iy in obstacle_actors]
            else:
                obstacle_actors = []

            agent = BasicAgent(env._vehicle, target_speed=args.target_speed)
            agent.set_destination(goal_loc)

            context_buffer: deque = deque(maxlen=args.context_len)
            context_start_snapshot_buffer: deque = deque(maxlen=args.context_len)
            stable_ticks = 0
            last_anchor_tick = -10**9
            ep_collisions = 0
            ep_anchors = 0
            current_obs_idx = 0
            passed_obstacles: set[int] = set()
            redirect_done: set[int] = set()

            for tick_idx in range(args.episode_len):
                if anchors_done >= target_anchors:
                    break

                if mode == "obstacle":
                    loc_pre = env._vehicle.get_location()
                    yaw_rad_pre = math.radians(env._vehicle.get_transform().rotation.yaw)
                    for oi, (ox, oy) in enumerate(obs_positions):
                        if oi in passed_obstacles:
                            continue
                        dx, dy = ox - loc_pre.x, oy - loc_pre.y
                        fwd_proj = dx * math.cos(yaw_rad_pre) + dy * math.sin(yaw_rad_pre)
                        if fwd_proj < -15.0:
                            passed_obstacles.add(oi)
                            if oi == current_obs_idx:
                                current_obs_idx += 1
                                agent.set_destination(goal_loc)

                    if (
                        current_obs_idx < len(obs_positions)
                        and current_obs_idx not in passed_obstacles
                        and current_obs_idx not in redirect_done
                    ):
                        ox, oy = obs_positions[current_obs_idx]
                        dx, dy = ox - loc_pre.x, oy - loc_pre.y
                        dist = math.sqrt(dx**2 + dy**2)
                        fwd_proj = dx * math.cos(yaw_rad_pre) + dy * math.sin(yaw_rad_pre)
                        if fwd_proj > 0 and dist < args.obstacle_redirect_distance:
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
                                        agent.set_destination(ahead[0].transform.location)
                                        redirect_done.add(current_obs_idx)

                control = agent.run_step()
                expert_action = carla_to_action(control.steer, control.throttle, control.brake)
                nominal_noise = rng.normal(0.0, np.asarray(args.nominal_noise_sigma, dtype=np.float32))
                nominal_action = np.clip(expert_action + nominal_noise, -1.0, 1.0).astype(np.float32)

                # Snapshot BEFORE applying nominal_action. This snapshot is aligned
                # with the resulting frame in context_buffer and is used as the
                # context-start state for branch replay.
                pre_step_snapshot = make_world_snapshot(env, obstacle_actors)

                obs, info = env.step(nominal_action)
                img, state = obs
                meta = compute_meta_for_mode(
                    mode=mode,
                    env=env,
                    obstacle_actors=obstacle_actors,
                    image_size=args.image_size,
                    visible_distance_threshold=args.visible_distance_threshold,
                    visible_bbox_area_threshold=args.visible_bbox_area_threshold,
                )
                lane_id, lat_dev = get_lane_info(env, carla_map)

                if float(state[0]) >= args.speed_threshold_mps:
                    stable_ticks += 1
                else:
                    stable_ticks = 0

                frame = make_frame(
                    image=img,
                    state=state,
                    action=nominal_action,
                    expert_action=expert_action,
                    lane_id=lane_id,
                    lateral_dev=lat_dev,
                    meta=meta,
                    step_collision=bool(info.get("collision", False)),
                    step_lane_invasion=bool(info.get("lane_invasion", False)),
                )
                context_buffer.append(frame)
                context_start_snapshot_buffer.append(pre_step_snapshot)

                if info.get("collision", False):
                    ep_collisions += 1
                    break

                if should_take_anchor(
                    mode=mode,
                    context_len=args.context_len,
                    context_buffer=context_buffer,
                    stable_ticks=stable_ticks,
                    stable_speed_ticks=args.stable_speed_ticks,
                    tick_idx=tick_idx,
                    last_anchor_tick=last_anchor_tick,
                    anchor_stride=args.anchor_stride,
                    meta=meta,
                    min_obstacle_distance=args.min_obstacle_distance,
                    max_obstacle_distance=args.max_obstacle_distance,
                ):
                    context_frames = list(context_buffer)
                    context_start_snapshot = context_start_snapshot_buffer[0]
                    snapshot = make_snapshot(
                        env=env,
                        obstacle_actors=obstacle_actors,
                        anchor_image=img,
                        anchor_state=state,
                        anchor_expert_action=expert_action,
                        anchor_nominal_action=nominal_action,
                        anchor_meta=meta,
                        anchor_lane_id=lane_id,
                        anchor_lateral_dev=lat_dev,
                    )
                    steer_samples = sample_steers(
                        rng,
                        args.steer_min,
                        args.steer_max,
                        args.steer_bins,
                        args.samples_per_bin,
                    )
                    anchor_branch_diags = []
                    for bin_id, steer in steer_samples:
                        cf_action = np.array([steer, float(expert_action[1])], dtype=np.float32)
                        replayed_context_frames, branch_frames, branch_diag, branch_anchor_snapshot = (
                            collect_branch_with_context_replay(
                                env=env,
                                carla_map=carla_map,
                                context_start_snapshot=context_start_snapshot,
                                context_frames_nominal=context_frames,
                                cf_action=cf_action,
                                mode=mode,
                                obstacle_actors=obstacle_actors,
                                image_size=args.image_size,
                                branch_horizon=args.branch_horizon,
                                diagnostic_horizon=args.diagnostic_horizon,
                                visible_distance_threshold=args.visible_distance_threshold,
                                visible_bbox_area_threshold=args.visible_bbox_area_threshold,
                            )
                        )
                        if (
                            len(replayed_context_frames) != args.context_len
                            or len(branch_frames) != args.branch_horizon
                            or branch_anchor_snapshot is None
                        ):
                            continue
                        accum.append_sequence(
                            episode_id=global_episode_id,
                            anchor_id=global_anchor_id,
                            branch_id=global_branch_id,
                            bin_id=bin_id,
                            sampled_steer=steer,
                            context_frames=replayed_context_frames,
                            branch_frames=branch_frames,
                            task_label=0 if mode == "clean" else 1,
                            target_speed=args.target_speed,
                            nominal_noise_sigma=tuple(args.nominal_noise_sigma),
                            branch_diag=branch_diag,
                            anchor_snapshot=branch_anchor_snapshot,
                        )
                        row = {
                            "mode": mode,
                            "anchor_id": global_anchor_id,
                            "branch_id": global_branch_id,
                            "bin_id": bin_id,
                            "steer": steer,
                            **branch_diag,
                        }
                        diag_rows.append(row)
                        anchor_branch_diags.append(row)
                        global_branch_id += 1
                        global_episode_id += 1

                    anchors_done += 1
                    ep_anchors += 1
                    last_anchor_tick = tick_idx
                    global_anchor_id += 1
                    restore_snapshot(env, snapshot)

                    if anchor_branch_diags:
                        max10 = np.mean([r["max_abs_cte_delta_10"] for r in anchor_branch_diags])
                        maxdiag = np.mean([r["max_abs_cte_delta_diag"] for r in anchor_branch_diags])
                        inv10 = np.mean([float(r["lane_invasion_10"]) for r in anchor_branch_diags])
                        col10 = np.mean([float(r["collision_10"]) for r in anchor_branch_diags])
                    else:
                        max10 = maxdiag = inv10 = col10 = float("nan")
                    print(
                        f"  Anchor {anchors_done:3d}/{target_anchors} [{mode}] "
                        f"global={global_anchor_id - 1} speed={float(state[0]):.2f} "
                        f"obs_vis={bool(meta['obstacle_visible'])} "
                        f"obs_dist={float(meta['obstacle_distance']):.1f} "
                        f"mean_max_cte_10={max10:.3f} "
                        f"mean_max_cte_diag={maxdiag:.3f} "
                        f"lane_inv10={inv10:.2f} col10={col10:.2f}"
                    )

                if goal_distance(env._vehicle, goal_loc) < 15.0 or agent.done():
                    break

            print(f"  Episode attempt={attempts:3d} mode={mode} anchors={ep_anchors} collisions={ep_collisions}")

    finally:
        from active_inference.evaluation.obstacles import destroy_obstacles
        destroy_obstacles(obstacle_actors)

    return global_anchor_id, global_branch_id, global_episode_id, summarize_diag(diag_rows, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean/obstacle counterfactual data.")
    parser.add_argument("--town", default="Town06_Opt")
    parser.add_argument("--route_key", default="Town06_Opt_TaskB")
    parser.add_argument("--mode", choices=["clean", "obstacle", "both"], default="both")
    parser.add_argument("--output", default="data/counterfactual_v6_smoke.h5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clean_anchors", type=int, default=5)
    parser.add_argument("--obstacle_anchors", type=int, default=5)
    parser.add_argument("--context_len", type=int, default=40)
    parser.add_argument("--branch_horizon", type=int, default=10)
    parser.add_argument("--diagnostic_horizon", type=int, default=20)
    parser.add_argument("--steer_min", type=float, default=-0.2)
    parser.add_argument("--steer_max", type=float, default=0.2)
    parser.add_argument("--steer_bins", type=int, default=8)
    parser.add_argument("--samples_per_bin", type=int, default=3)
    parser.add_argument("--target_speed", type=float, default=25.0)
    parser.add_argument("--speed_threshold_mps", type=float, default=5.5)
    parser.add_argument("--stable_speed_ticks", type=int, default=20)
    parser.add_argument("--anchor_stride", type=int, default=20)
    parser.add_argument("--nominal_noise_sigma", type=float, nargs=2, default=[0.02, 0.02])
    parser.add_argument("--num_obstacles", type=int, default=2)
    parser.add_argument("--obstacle_redirect_distance", type=float, default=60.0)
    parser.add_argument("--min_obstacle_distance", type=float, default=10.0)
    parser.add_argument("--max_obstacle_distance", type=float, default=50.0)
    parser.add_argument("--visible_distance_threshold", type=float, default=50.0)
    parser.add_argument("--visible_bbox_area_threshold", type=float, default=20.0)
    args = parser.parse_args()

    if args.diagnostic_horizon < args.branch_horizon:
        raise ValueError("diagnostic_horizon must be >= branch_horizon")

    try:
        import carla  # noqa: F401
        from agents.navigation.basic_agent import BasicAgent  # noqa: F401
    except ImportError:
        print("ERROR: carla package and CARLA PythonAPI agents are required.")
        sys.exit(1)

    from active_inference.data.carla_env import CARLADrivingEnv

    rng = np.random.default_rng(args.seed)
    accum = H5Accumulator()
    env = CARLADrivingEnv(host=args.host, port=args.port, town=args.town, image_model_size=args.image_size)

    try:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)
        carla_map = client.get_world().get_map()

        print("=" * 100)
        print("Collect counterfactual v6")
        print(f"town: {args.town}")
        print(f"route_key: {args.route_key}")
        print(f"mode: {args.mode}")
        print(f"output: {args.output}")
        print(f"clean_anchors: {args.clean_anchors}")
        print(f"obstacle_anchors: {args.obstacle_anchors}")
        print(f"context_len: {args.context_len}")
        print(f"branch_horizon: {args.branch_horizon}")
        print(f"diagnostic_horizon: {args.diagnostic_horizon}")
        print("replay_context_for_each_branch: True")
        print(f"steer range: [{args.steer_min}, {args.steer_max}]")
        print(f"steer_bins: {args.steer_bins}")
        print(f"samples_per_bin: {args.samples_per_bin}")
        print(f"branches_per_anchor: {args.steer_bins * args.samples_per_bin}")

        anchor_id = 0
        branch_id = 0
        episode_id = 0
        summaries: dict[str, Any] = {}

        if args.mode in ("clean", "both") and args.clean_anchors > 0:
            anchor_id, branch_id, episode_id, summary = collect_mode(
                args=args,
                mode="clean",
                target_anchors=args.clean_anchors,
                env=env,
                carla_map=carla_map,
                client=client,
                rng=rng,
                accum=accum,
                global_anchor_start=anchor_id,
                global_branch_start=branch_id,
                global_episode_start=episode_id,
            )
            summaries.update(summary)

        if args.mode in ("obstacle", "both") and args.obstacle_anchors > 0:
            anchor_id, branch_id, episode_id, summary = collect_mode(
                args=args,
                mode="obstacle",
                target_anchors=args.obstacle_anchors,
                env=env,
                carla_map=carla_map,
                client=client,
                rng=rng,
                accum=accum,
                global_anchor_start=anchor_id,
                global_branch_start=branch_id,
                global_episode_start=episode_id,
            )
            summaries.update(summary)

        attrs = {
            "collection_type": "counterfactual_v6",
            "town": args.town,
            "route_key": args.route_key,
            "mode": args.mode,
            "seed": int(args.seed),
            "camera_image_size": int(args.image_size),
            "context_len": int(args.context_len),
            "branch_horizon": int(args.branch_horizon),
            "diagnostic_horizon": int(args.diagnostic_horizon),
            "sequence_len": int(args.context_len + args.branch_horizon),
            "replay_context_for_each_branch": True,
            "steer_min": float(args.steer_min),
            "steer_max": float(args.steer_max),
            "steer_bins": int(args.steer_bins),
            "samples_per_bin": int(args.samples_per_bin),
            "branches_per_anchor": int(args.steer_bins * args.samples_per_bin),
            "clean_anchors_requested": int(args.clean_anchors),
            "obstacle_anchors_requested": int(args.obstacle_anchors),
            "anchors_collected": int(anchor_id),
            "branches_collected": int(branch_id),
            "total_frames": int(len(accum.data["images"])),
            "target_speed": float(args.target_speed),
            "speed_threshold_mps": float(args.speed_threshold_mps),
            "stable_speed_ticks": int(args.stable_speed_ticks),
            "anchor_stride": int(args.anchor_stride),
            "nominal_noise_sigma_steer": float(args.nominal_noise_sigma[0]),
            "nominal_noise_sigma_accel": float(args.nominal_noise_sigma[1]),
            "num_obstacles": int(args.num_obstacles),
            "min_obstacle_distance": float(args.min_obstacle_distance),
            "max_obstacle_distance": float(args.max_obstacle_distance),
            "visible_distance_threshold": float(args.visible_distance_threshold),
            "visible_bbox_area_threshold": float(args.visible_bbox_area_threshold),
        }
        attrs.update(summaries)
        accum.write(args.output, attrs)

        print("=" * 100)
        print(f"Saved: {args.output}")
        print(f"anchors_collected: {anchor_id}")
        print(f"branches_collected: {branch_id}")
        print(f"frames: {len(accum.data['images'])}")
        for k, v in summaries.items():
            print(f"{k}: {v}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
# Evaluation v4: Route Redesign, Per-Frame Logging, EFE Exposure

**Date:** 2026-03-20
**Status:** Implemented

## Problem Statement

The Deep Active Inference autonomous driving agent has no navigation capability (no waypoint input, no directional preference). Evaluation v3 revealed:

1. **Town03 (Task B) is unfair**: Urban grid with 46 intersections over 400m — the agent cannot decide left/right/straight at intersections without waypoint guidance
2. **Task B should test obstacle avoidance only**: On intersection-free roads with static obstacles
3. **Evaluation logging is insufficient**: No per-frame position/heading data for map-based trajectory visualization
4. **EFE/epistemic scores are discarded**: Computed inside planner but never exposed to logging

## Design Decisions

### Why Town06_Opt for Task B (not Town04)

- Agent trained on Town06 data — no visual domain gap, isolates obstacle avoidance as the only variable
- Town06_Opt has wide multi-lane highways ideal for lane changes around obstacles
- Town04 stays dedicated to Task A (cross-domain curve following)

### Why Town03 was removed

- 46 intersections in 400m requires navigation decisions (left/right/straight)
- Agent has no waypoint input or directional preference mechanism
- Cannot fairly evaluate obstacle avoidance when navigation is the bottleneck

### PlanResult pattern

- `iCEMPlanner.plan()` now returns `PlanResult(action, efe_score, epistemic_score)`
- One extra rollout of elite mean (1 sample vs 200 in CEM — negligible cost)
- `agent.step()` remains backward-compatible via delegation to `step_with_info()`

## Changes Made

### Stage 1: EFE/Epistemic Exposure
- Added `PlanResult` NamedTuple to `cem_planner.py`
- `plan()` returns `PlanResult` with EFE and epistemic scores
- `agent.step_with_info()` returns full `PlanResult`
- `agent.step()` delegates to `step_with_info().action` for backward compatibility

### Stage 2: Route Redesign
- Removed Town03 entries from `EVAL_ROUTES`
- Added `Town06_Opt_TaskB` route key (placeholder, finalized via route discovery)
- Added `find_straight_routes()` for discovering low-turn multi-lane segments
- Updated config defaults: `task_b: Town06_Opt`

### Stage 3: Obstacle Spawning Module
- New `src/active_inference/evaluation/obstacles.py`
- Extracted from `scripts/collect_preference_data.py` lines 70-89
- `spawn_obstacles()` / `destroy_obstacles()` — reusable across eval and data collection

### Stage 4: Evaluation Script Overhaul
- Separate `TOWN_MAP` (which CARLA map to load) from `ROUTE_MAP` (which routes to use)
- Uses `step_with_info()` for EFE/epistemic logging
- Spawns obstacles for Task B; cleans up in finally block
- Per-frame JSONL logging: 19 fields per frame including x/y/z, yaw, EFE, epistemic
- Updated CSV with `mean_efe_score`, `mean_epistemic_score`, `trajectory_file` columns
- Default output: `outputs/eval_v4`

### Stage 5: Route Discovery Utility
- `scripts/discover_routes.py` CLI for scanning spawn-point pairs
- Supports both `obstacle_avoidance` (low-turn) and `curves` (high-turn) modes

## Per-Frame JSONL Schema

| Field | Source | Purpose |
|-------|--------|---------|
| frame_id | counter | time index |
| timestamp | frame_id * 0.05 | simulation time |
| x, y, z | vehicle.get_location() | map trajectory |
| yaw | vehicle.get_transform().rotation.yaw | heading |
| speed_mps | state[0] | velocity |
| steer | state[1] | current control |
| action_steer, action_accel | agent output | commanded control |
| lateral_dev | waypoint distance | lane deviation |
| lane_id, road_id | CARLA waypoint | lane tracking |
| collision, lane_invasion | env info flags | safety events |
| efe_score | PlanResult | planning quality |
| epistemic_score | PlanResult | uncertainty |
| route_completion_pct | route progress | navigation |
| distance_to_goal | Euclidean | proximity |

File naming: `trajectory_{task}_r{route}_ep{episode}.jsonl`

## Pending

- Run `scripts/discover_routes.py` with CARLA to finalize Town06_Opt Task B routes
- Hardcode discovered routes into `EVAL_ROUTES["Town06_Opt_TaskB"]`

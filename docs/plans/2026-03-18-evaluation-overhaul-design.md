# Evaluation Overhaul Design

## Problem Statement

The current evaluation system has fundamental flaws:
1. **No goal-based success**: "Success" was defined as "no collision for 1000 frames" — a stationary car passes
2. **No fixed routes**: Random spawn points, no destination, no curvature requirement
3. **No obstacle avoidance metric**: Lane invasions counted but no maneuver-level success tracking
4. **First-person only**: No external viewpoint to verify driving behavior

## Design Decisions

### 1. Route-Based Evaluation

**Routes** are defined as (start_spawn_index, goal_spawn_index) pairs per town. Each route:
- Has **sufficient distance** (>200m)
- Contains **at least 2 curves** (turns detected via GlobalRoutePlanner's RoadOption.LEFT/RIGHT)
- Is verified at startup by tracing the route and counting turn segments

**Success criteria**:
- Vehicle reaches within 10m of goal location
- Vehicle does not get stuck (speed > 0.5 m/s on average over any 5-second window)
- No sustained collision (collision that stops the vehicle for >3 seconds)

**Failure conditions**:
- Collision that halts progress (stuck for >3s)
- Off-road for >5 consecutive seconds
- Timeout (max_frames reached without reaching goal)

### 2. Task Definitions

**Task A (Lane Keeping with Curves)**: Town04
- Route with curves, no lane changes required
- Goal: reach destination staying in lane
- Metrics: success, route_completion_pct, mean_lateral_dev, frames

**Task B (Obstacle Avoidance)**: Town03
- Route that requires at least one lane change to avoid obstacle
- Goal point is in the CHANGED lane (so no return-to-original-lane needed)
- Obstacle avoidance sub-metric: did the vehicle change lanes without collision?
- Metrics: success, route_completion_pct, avoidance_success, mean_lateral_dev, frames

**Baseline**: Town06_Opt (training distribution)
- Standard route on familiar map
- Metrics: same as Task A

### 3. Image Cropping

Store full 64x64 images in training data. During training, optionally crop to bottom 60% (rows 26-64) to focus on road surface and nearby objects, then resize back to 64x64. Controlled by config flag `encoder.crop_road: true/false`.

The crop removes sky/distant buildings that vary across towns but are irrelevant to driving.

### 4. Chase Camera Visualization

Use CARLA's spectator actor positioned behind and above the ego vehicle:
- Offset: x=-8m (behind), z=5m (above), pitch=-15 degrees
- Updated every simulation tick to follow vehicle
- Recorded via a second RGB camera sensor attached with the same transform
- Saved as MP4 at 20 FPS

### 5. Metrics Schema

CSV output fields:
```
task, town, episode, route_id, success, route_completion_pct,
mean_lateral_dev, offroad_events, avoidance_success, frames, goal_distance
```

## Implementation Phases

1. **Route system** (`src/active_inference/evaluation/routes.py`): Define routes per town, validate curves
2. **Evaluation script** (`scripts/evaluate.py`): Rewrite with route tracking, new success criteria
3. **Chase camera** (`src/active_inference/data/carla_env.py`): Add spectator/chase camera
4. **Image cropping** (`src/active_inference/utils/transforms.py` + config): Add crop_road option
5. **Additional training**: If needed based on crop evaluation
6. **Final evaluation + video**: Run all tasks, record chase-cam videos

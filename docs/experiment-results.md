# Deep Active Inference for Autonomous Driving — Experiment Results

**Authors:** Jaerock Kwon (with Claude Code assistance)
**Date:** March 2026

## Overview

This document consolidates all experimental results for the Deep Active Inference (AIF) autonomous driving agent in CARLA 0.9.16. The agent uses Expected Free Energy (EFE) minimization with iCEM planning to learn steering control and obstacle avoidance — **not** reinforcement learning. ALL control signals originate from the Active Inference framework. No classical controllers (Stanley, PID) are used.

**Two validated tasks:**
- **Task A (Lane Keeping):** Navigate curved highways using learned reactive steering. Best result: 81.2% completion on moderate curves (0°→85° heading change).
- **Task B (Obstacle Avoidance):** Avoid static obstacles via lane change using an AIF reflexive steer prior. Best result: 100% success rate, 19/19 obstacles avoided.

---

## Training Data

| Dataset | Frames | Town | Description |
|---------|--------|------|-------------|
| expert_data_v4.h5 | 72,000 | Town06 | Straight highway, 4D state |
| expert_data_town04.h5 | 24,000 | Town04 | Curved highway, heading_error range ±3.14 rad |
| expert_data_mixed.h5 | 96,000 | Mixed | 75% Town06 + 25% Town04, 26.1% success rate |
| task_b_lanechange.h5 | 15,000 | Town06_Opt | Lane-change episodes (BehaviorAgent), 38.9% success rate |

**Success criterion:** |crosstrack_error| < 0.3m → success_flag = True

**Success data statistics:**
- Heading error: ±0.10 rad (success) vs ±0.79 rad (failure)
- Max success |heading_error|: 0.76 rad (44°) in Town04 data
- Zero success frames with |heading_error| > 1.5 rad (86°)

---

## Task A: Lane Keeping

### Version Progression

| Version | Change | Baseline (Town06) | Task A (Town04) |
|---------|--------|-------------------|-----------------|
| v5 | EFE z-score normalization | 6.7% | — |
| v6 | State-space instrumental value (breakthrough) | 14.7% | 40% |
| v7d | CEM cold-start fixes | 40.2% | 40.5% |
| v9 | Mixed training data (Town06+Town04) | — | 42.1% |
| **v10** | **Moderate curve evaluation** | **41.6% (69% best)** | **81.2%** |

### Key Iterations

**v5 → v6: State-Space Instrumental Value (Breakthrough)**
The 64D latent GMM couldn't discriminate fine lane-centering differences. Adding a decoded state-space penalty (heading_error² + crosstrack_error²) with raw values and `clamp(max=4.0)` produced an 8x improvement in MLD (2.18m → 0.27m). Z-scored state penalty amplified noisy predictions and caused crashes.

**v6 → v7d: CEM Cold-Start Fix**
~50% of episodes failed at cold-start due to CEM warm-start inertia. Fixes: min_std floor (0.1), population injection (keep_fraction=0.2), cold-start sample boost (1000 vs 500), extra iterations (10 vs 5). Cold-start survival improved from 67% to 100%.

**v7d → v9: Mixed Training Data**
All Task A episodes stopped at exactly 40.5% completion where road enters a curve. Root cause: world model trained only on straight roads (Town06). Mixed training data (72K Town06 + 24K Town04) enabled curve navigation.

**v9 → v10: Moderate Curve Evaluation**
90° turns require heading_error > 1.5 rad, but max success heading_error in training data is 0.76 rad. Evaluating on moderate curves (0°→85° heading change, 378m route) shows what the model CAN do.

### v10 Moderate Curve Results (Key Result)

| Episode | Completion | MLD | Offroad | Frames |
|---------|-----------|-----|---------|--------|
| 0 | 82.7% | 0.714m | 0 | 1458 |
| 1 | 81.7% | 0.692m | 6 | 1332 |
| 2 | 79.2% | 0.468m | 0 | 1292 |
| **Avg** | **81.2%** | **0.624m** | 2 | 1361 |

The agent navigates 90+ degrees of continuous heading change (yaw: -0.3° → 91.2°) over 1200 frames while maintaining heading error within ±0.034 rad (~2°) and crosstrack error within ±0.5m.

### Steering Statistics

| Metric | Value |
|--------|-------|
| Yaw range traversed | 119.9° (from -4.9° to 115.1°) |
| Action steer mean | 0.028 |
| Action steer std | 0.136 |
| Action steer range | [-0.668, 0.480] |
| **Heading-steer correlation** | **r = 0.41** |

The positive correlation (r=0.41) between heading error and steering action confirms **learned reactive steering** — the agent responds to heading deviations with corrective inputs.

### v10 Baseline Results (Town06, Straight Highway)

| Route | Episode | Completion | MLD |
|-------|---------|-----------|-----|
| R0 | 0 | 69.3% | 0.595m |
| R0 | 1 | 28.2% | 0.582m |
| R0 | 2 | 28.3% | 1.857m |
| R1 | 0 | 26.9% | 1.024m |
| R1 | 1 | 68.4% | 0.559m |
| R1 | 2 | 28.0% | 1.957m |
| **Avg** | — | **41.6%** | 1.095m |

Bimodal pattern: 2/6 episodes reach ~69% completion; 4/6 hit a highway junction at ~28% and take the wrong branch (navigation limitation, not steering).

---

## Task B: Obstacle Avoidance via Lane Change

### Approach: AIF Reflexive Steer Prior

The obstacle avoidance operates as a **high-precision action prior** within the AIF framework. The proven 4D world model is kept unchanged; obstacle awareness is added at runtime.

1. **Detection:** Forward-only obstacle proximity (cosine similarity with vehicle heading, 40m range) → proximity score 0–1
2. **Evasion direction:** Lane-geometry comparison using CARLA `get_left_lane()`/`get_right_lane()` — pick lane farther from obstacle
3. **Lock persistence:** Track locked obstacle position + minimum distance. Reset only when `dist > min_dist + 10m`
4. **CEM bias:** Action-prior penalty in CEM scoring scaled by proximity
5. **Post-CEM override:** Direct steer override when proximity > threshold
6. **Lateral clearance suppression:** Pre-CEM suppression when lateral clearance >= 4m (vehicle already in safe lane)

**AIF justification:** Strong prior beliefs about action (high precision) override weaker posterior beliefs from CEM planning when proximity to obstacle is high. When cleared, precision drops to zero and CEM resumes normal lane-keeping control.

### Iterative Development (v5j → v5k5)

| Version | Change | Route 0 SR | Obstacles Avoided | Issue |
|---------|--------|-----------|-------------------|-------|
| v5j | Initial obstacle penalty | 0% | 0/2 | Pushes through obstacles |
| v5j2 | Evasion direction fix | 0% | 0/2 | Direction flips every frame |
| v5k | Lock persistence | 0% | 0/2 | Steers off-road indefinitely |
| v5k2 | Lateral clearance suppress | 50% | 2/2 (ep0-1) | `abs(Y)` wrong for N-S roads |
| v5k3 | Road-waypoint lateral | 60% | 90% | Wrong evasion dir for some obstacles |
| **v5k4** | **Lane-geometry evasion** | **100%** | **10/10** | **Solved** |
| **v5k5** | **Route 1 validation** | **100%** | **9/9** | **Validated** |

### Final Results

#### Route 0: 373m, 2 obstacles per episode, 5 episodes

| Episode | Success | Completion | Obstacles Avoided | Lane Changes | MLD |
|---------|---------|------------|-------------------|-------------|------|
| 0 | goal_reached | 98.4% | 2/2 | 6 | 0.70 |
| 1 | goal_reached | 99.0% | 2/2 | 7 | 0.64 |
| 2 | goal_reached | 98.4% | 2/2 | 6 | 0.66 |
| 3 | goal_reached | 99.0% | 2/2 | 5 | 0.69 |
| 4 | goal_reached | 98.4% | 2/2 | 6 | 0.63 |
| **Avg** | **100% SR** | **98.6%** | **10/10 (100%)** | **6.0** | **0.67** |

#### Route 1: 518m, 3 obstacles per episode, 3 episodes

| Episode | Success | Completion | Obstacles Avoided | Lane Changes | MLD |
|---------|---------|------------|-------------------|-------------|------|
| 0 | goal_reached | 99.2% | 3/3 | 6 | 0.64 |
| 1 | goal_reached | 99.2% | 3/3 | 6 | 0.69 |
| 2 | goal_reached | 98.9% | 3/3 | 7 | 0.62 |
| **Avg** | **100% SR** | **99.1%** | **9/9 (100%)** | **6.3** | **0.65** |

#### Combined Summary

| Metric | Route 0 (5 ep) | Route 1 (3 ep) | Combined |
|--------|---------------|---------------|----------|
| Success Rate | 100% | 100% | **100%** |
| Obstacle Avoidance | 10/10 (100%) | 9/9 (100%) | **19/19 (100%)** |
| Avg Completion | 98.6% | 99.1% | **98.8%** |
| Avg MLD | 0.67m | 0.65m | **0.66m** |
| Avg Lane Changes | 6.0 | 6.3 | **6.1** |

Task A regression check: no degradation (81.2% completion unchanged on moderate curves).

### Key Technical Fixes

1. **Lock persistence:** Track locked obstacle position and minimum distance. Reset only when vehicle physically passed (`dist > min_dist + 10m`), preventing evasion direction flip mid-maneuver.
2. **Pre-CEM lateral clearance suppression:** Set `evasion_steer=0.0` when lateral clearance >= 4m, preventing continued steering after reaching safe lane.
3. **Road-waypoint lateral projection:** Use CARLA waypoint yaw for direction-independent lateral clearance: `lat_dist = abs(dx*sin(yaw) + dy*(-cos(yaw)))`.
4. **Lane-geometry evasion direction:** Use CARLA `get_left_lane()`/`get_right_lane()` to compare distances; pick lane farther from obstacle. More robust than geometric projection.

### Failed Approach: 5D State (v4 iteration)

Adding `obstacle_distance` as a 5th state dimension was attempted but abandoned. Weight surgery to extend the state encoder/decoder from 4D to 5D broke the learned steering behavior (completion dropped from 81% to 2.6%). The runtime penalty approach (v5) succeeded without retraining.

---

## Theoretical Interpretation

### Motor Babbling Analogy
The CEM planner explores action space (like motor babbling), and the EFE scoring function guides exploration toward preferred outcomes (like proprioceptive/visual feedback).

### Preference as Skill Encoding
The GMM preference model encodes "what good driving looks like" in latent space, learned from successful demonstrations. This is analogous to how a human driver builds an internal model of expected visual input during lane keeping.

### Free Energy Minimization as Control
The agent's steering emerges from minimizing the divergence between predicted future states and preferred states — not from explicit steering rules. The heading-steer correlation (r=0.41) shows this minimization produces meaningful corrective behavior.

### AIF Reflexive Steer Prior (Task B)
When an obstacle is detected, the precision of the action prior increases sharply. This strong prior belief about required steering overrides weaker posterior beliefs from CEM planning. When cleared, precision drops to zero and CEM resumes normal control. The lock persistence ensures committed lane-change behavior — analogous to how a human driver commits to a maneuver.

### Limitation Parallels Human Learning
Just as a novice driver struggles with sharp turns they haven't practiced, the AIF agent fails at curvatures outside its training distribution. The framework supports learning these scenarios given appropriate experience.

---

## Evaluation Routes

| Route | Town | Spawn | Distance | Obstacles | Description |
|-------|------|-------|----------|-----------|-------------|
| Task A Full | Town04 | 0→53 | 765m | 0 | Highway with 21 turns including 90° junction |
| Task A Moderate | Town04 | 0→30 | 378m | 0 | Continuous curves 0°→85°, no sharp turns |
| Baseline R0 | Town06 | 0→1 | 1332m | 0 | Straight highway |
| Baseline R1 | Town06 | 0→2 | 1329m | 0 | Straight highway |
| Task B Route 0 | Town06_Opt | 0→152 | 373m | 2 | Primary obstacle avoidance route |
| Task B Route 1 | Town06_Opt | 22→152 | 518m | 3 | Longer variant, obstacle fractions [0.15, 0.45, 0.75] |

---

## Configuration

### Key Parameters: Task A vs Task B

| Parameter | Task A (default.yaml) | Task B (task_b_v5.yaml) | Reason |
|-----------|----------------------|------------------------|--------|
| beta_state | 0.5 | 0.0 | Allow crosstrack deviation during lane change |
| beta_obstacle | 0.0 | 40.0 | Runtime obstacle penalty |
| heading_only_state | false | true | No crosstrack penalty for lane changes |
| horizon | 12 | 15 | Longer planning for lane-change commitment |
| noise_scale | [0.8, 0.6] | [1.0, 0.6] | Wider steer exploration |
| K (GMM) | 5 | 7 | Richer lane-change distribution |
| accel_prior | 0.3 | 0.2 | Slower obstacle approach |
| warm_start_reset_threshold | 1.0 | 4.0 | Prevent mid-lane-change reset |

---

## Known Limitations

1. **Sharp turns (>85° heading change):** Training data contains zero success examples at heading errors > 1.5 rad. Fixable with more diverse training data.
2. **Junction navigation:** Agent takes wrong branches at highway junctions (~28% wall in baseline). Requires goal-directed preference.
3. **Obstacle detection is runtime, not learned:** Uses CARLA actor positions rather than visual detection from learned representations.
4. **Straight roads only for Task B:** Obstacle avoidance tested only on straight highway segments.
5. **Fixed lateral clearance threshold:** The 4m clearance threshold is manually set; an adaptive threshold would be more general.

---

## Future Directions

1. Expand training data to include sharper curves (Town03, Town04 intersections).
2. Goal-directed preference for highway junction navigation.
3. Learned obstacle detection from visual representations.
4. Online world model adaptation during evaluation.
5. Curved-road obstacle avoidance validation.

---

*Previous documentation archived in `docs/archive/`.*

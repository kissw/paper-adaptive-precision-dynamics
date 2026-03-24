# Task B: Obstacle Avoidance via Lane Change — Progress Log

**Date:** 2026-03-24
**Objective:** Train an AIF agent that avoids obstacles by changing lanes (not pushing through).

## Architecture Decision: Two Models

- **Shared:** RSSM + encoder + decoders (trained on combined Task A + Task B data)
- **Separate:** GMM preference (Task A: lane-centered, Task B: lane-change) + EFE config
- **Rationale:** Crosstrack conflict — Task A penalizes crosstrack²→0; during lane change crosstrack shifts ±3.5m. Single model would need context-dependent preference switching.

## Pipeline Stages

### Stage 1: Code Implementation (DONE)

Files changed:
- `src/active_inference/evaluation/obstacles.py` — `spawn_obstacles_multilane()`, `spawn_obstacles_on_route()`
- `src/active_inference/planning/efe.py` — `state_instrumental_value_heading_only()`
- `src/active_inference/planning/cem_planner.py` — configurable `warm_start_reset_threshold`
- `src/active_inference/config.py` — `heading_only_state`, `warm_start_reset_threshold`
- `src/active_inference/agent.py` — pass new config fields through
- `configs/experiment/task_b.yaml` — Task B config (beta_state=0.2, horizon=15, K=7)
- `scripts/collect_task_b_data.py` — lane-change data collection
- `scripts/refit_preference.py` — `--task_filter B --K 7`
- `scripts/evaluate.py` — route-based obstacles, avoidance metrics, lane-change detection

Commit: `a7bd0b2` — 72/72 tests pass.

### Stage 2: Task B Data Collection (DONE)

- 15,000 frames collected across 36 episodes
- Agent: BehaviorAgent(aggressive) — reliable lane changes
- 80.6% of episodes contained lane changes (29/36)
- 38.9% success rate (14/36 — collision-free with lane changes)
- 88.1% of frames labeled as Task B (lane-change episodes)
- Tier distribution: clean=15, medium=11, high=10
- Output: `data/task_b_lanechange.h5`

### Stage 3: Merge Datasets (DONE)

Combined 96,000 + 15,000 = 111,000 frames → `data/expert_data_v6_combined.h5`

### Stage 4: Train Shared World Model (DONE)

- Fine-tuned from existing mixed-data checkpoint (`best.pt`)
- 20 epochs, ~30 min/epoch, ~10 hours total
- Loss: 1.5164 → 1.5147 (already near convergence from pre-trained checkpoint)
- GMM warning: `min_component_dist=0.23` — components clustering but functional
- Output: `outputs/train_v6_combined/checkpoints/best.pt`

### Stage 5: Refit Task B Preference (DONE)

- K=7 GMM fit on 5,000 latent samples from lane-change episodes
- Checkpoint K override: 5 (checkpoint) → 7 (target)
- GMM Health: log_prob=60.92, min_component_dist=0.0049
- Component weights: one dominant (72.7%), six minor (3-6%)
- EFE probe: accelerate=1113.9, brake=1131.6, zero=1149.8 (prefers driving)
- Output: `outputs/train_v6_combined/checkpoints/best_taskb_pref.pt`

### Stage 6: Evaluate Task B (DONE — v1)

#### Task B Results (Obstacle Avoidance)

| Route | Episode | Completion | MLD | Obstacles Avoided | Lane Changes | Termination |
|-------|---------|-----------|------|-------------------|--------------|-------------|
| 0 | 1 | 44% | 0.481 | 0/2 | 3 | collision_stuck |
| 0 | 2 | 44% | 0.448 | 0/2 | 3 | collision_stuck |
| 0 | 3 | 44% | 0.468 | 0/2 | 3 | collision_stuck |
| 1 | 1 | 68% | 0.548 | 0/3 | 5 | timeout |
| 1 | 2 | 79% | 1.938 | 0/3 | 81 | timeout |
| 1 | 3 | 67% | 0.540 | 0/3 | 4 | timeout |

**Summary:** SR=0%, Avg Completion=57.8%, Obstacles Avoided=0/15 (0%), Total Lane Changes=99

#### Task A Regression (Lane Keeping — Town04 moderate curves)

| Episode | Completion | MLD | Termination |
|---------|-----------|------|-------------|
| 1 | 100% | 2.017 | timeout |
| 2 | 79% | 0.374 | collision_stuck |
| 3 | 79% | 0.364 | collision_stuck |

**Summary:** SR=0%, Avg Completion=86.1%, Avg MLD=0.918

Videos: `outputs/eval_task_b_v1/*.mp4`, `outputs/eval_task_a_v6/*.mp4`

## Key Observations (v1 Iteration)

1. **Lane changes ARE happening** (99 total across 6 episodes, all episodes had at least 3)
2. **But NOT in response to obstacles** — 0/15 obstacles avoided
3. **Route 1 Ep 2 showed instability**: 81 lane changes (oscillation), MLD=1.938
4. **Task A not regressed significantly**: 86.1% avg completion (prev: 81.2% on v10)
5. **Core issue**: GMM preference captures "what lane-change driving looks like" but NOT "when to initiate lane changes near obstacles"

## Analysis: Why Obstacle Avoidance Fails

The preference model (GMM) is fit on latent states from lane-change episodes. These latents encode the visual + state information during lane changes, but:

1. **No obstacle-conditioned preference**: The GMM doesn't distinguish "near obstacle" vs "no obstacle" latent states. It simply captures the general distribution of lane-change driving.
2. **CEM plans in imagination space**: The iCEM planner imagines future trajectories through the world model, but the imagination doesn't include obstacle dynamics since obstacles are static objects not part of the learned state transition model.
3. **Temporal disconnect**: Lane changes happen at random times because the EFE is similar whether or not an obstacle is ahead — the preference model assigns equal probability to lane-change states regardless of context.

## Key Parameters (Task B vs Task A)

| Parameter | Task A | Task B | Reason |
|-----------|--------|--------|--------|
| beta_state | 0.5 | 0.2 | Allow crosstrack deviation |
| horizon | 12 | 15 | Longer planning for lane-change commitment |
| noise_scale | [0.8, 0.6] | [1.0, 0.6] | Wider steer exploration |
| K (GMM) | 5 | 7 | Richer lane-change distribution |
| accel_prior | 0.3 | 0.2 | Slower obstacle approach |
| warm_start_reset_threshold | 1.0 | 4.0 | Prevent mid-lane-change reset |

## Bug Fixes Applied

1. **task_b.yaml missing encoder section**: Config defaulted to `state_dim=2` instead of 4, causing `RuntimeError: size mismatch` when loading checkpoint. Fixed by adding `encoder` and `rssm` sections.
2. **Pipeline `set -e` masked by pipe**: `python ... | tee log` masks Python errors because `set -e` only checks the last pipe command (`tee`). Fixed with `set -eo pipefail`.
3. **Missing eval output directory**: `tee` failed because `outputs/eval_task_b_v1/` didn't exist. Fixed with `mkdir -p` at script start.
4. **Preference K mismatch on checkpoint load**: Checkpoint has K=5, task_b.yaml has K=7. `load_checkpoint()` failed on size mismatch. Fixed by probing checkpoint K first, building agent with checkpoint K, then reinitializing preference with target K.

## v2 Iteration: heading_only_state=true

Config: `beta_state=0.2`, `heading_only_state=true` (no crosstrack penalty)

| Route | Episode | Completion | MLD | Obstacles Avoided | Lane Changes | Termination |
|-------|---------|-----------|------|-------------------|--------------|-------------|
| 0 | 1 | 44% | 0.848 | 0/2 | 3 | collision_stuck |
| 0 | 2 | 44% | 0.774 | 0/2 | 3 | collision_stuck |
| 0 | 3 | 44% | 0.854 | 0/2 | 3 | collision_stuck |
| 1 | 1 | 77% | 0.868 | 0/3 | 3 | timeout |
| 1 | 2 | 66% | 0.971 | 0/3 | 6 | timeout |
| 1 | 3 | 66% | 0.893 | 0/3 | 4 | timeout |

**Summary:** SR=0%, Avg Completion=57.0%, Obstacles Avoided=0/15 (0%), Lane Changes=22
**Worse than v1** — fewer lane changes (22 vs 99), higher MLD, more offroad events.

## v3 Iteration: beta_state=0.0 (pure GMM preference)

Config: `beta_state=0.0`, `heading_only_state=true` (no state penalty at all)

| Route | Episode | Completion | MLD | Obstacles Avoided | Lane Changes | Termination |
|-------|---------|-----------|------|-------------------|--------------|-------------|
| 0 | 1 | 44% | 0.576 | 0/2 | 3 | collision_stuck |

**Partial (killed after confirming same pattern):** Same collision with obstacles, 0 avoidance.

## Analysis: Why Parameter Tuning Fails

All three iterations (v1 beta_state=0.2, v2 heading_only, v3 beta_state=0.0) show **identical failure**: 0 obstacles avoided despite 22-99 lane changes. The root cause is structural, not parametric:

1. **GMM preference is obstacle-blind**: Fit on all lane-change frames, it captures "general lane-change driving" distribution. No discrimination between "near obstacle" and "no obstacle" latent states.
2. **State decoder has no obstacle signal**: 4D state [speed, steer, heading, crosstrack] contains no obstacle proximity information. EFE penalty cannot reference what the model doesn't predict.
3. **RSSM imagination can't plan around obstacles**: Without obstacle distance in the decoded state, the CEM planner has no way to prefer "lane change near obstacle" over "lane change elsewhere."

## v4: 5D State Approach (IN PROGRESS)

### Design (stays within AIF framework)

Add `obstacle_distance` as 5th state dimension:
- **State vector**: [speed, steer, heading_error, crosstrack_error, **obstacle_distance_norm**]
- **Normalization**: `min(distance_to_nearest_obstacle / 50.0, 1.0)` → 0=at obstacle, 1=far away
- **EFE penalty**: `beta_obstacle * (1 - decoded_obstacle_dist)^2` — penalizes imagined trajectories near obstacles
- **Weight surgery**: Load existing 4D checkpoint into 5D model by padding new dimension weights with zeros
- **Task A compatibility**: Pad Task A data with obstacle_distance=1.0 (no obstacles)

### AIF Justification

This is equivalent to adding a **prior preference for obstacle-free states** in the AIF framework:
- The state decoder learns to predict obstacle proximity from RSSM features
- During CEM imagination, decoded obstacle_distance serves as a direct planning signal
- The obstacle proximity penalty in EFE = negative log-preference for obstacle-near states
- The agent learns to **prefer** trajectories that keep obstacle distance high

### Implementation

Files changed:
- `src/active_inference/config.py` — added `beta_obstacle` to EFEConfig
- `src/active_inference/planning/efe.py` — `obstacle_proximity_penalty()` method + integration in `score()`
- `src/active_inference/agent.py` — weight surgery in `load_checkpoint()` for state_dim changes, preference K mismatch handling
- `scripts/collect_task_b_data.py` — compute obstacle_distance per frame, error recovery for CARLA crashes
- `scripts/evaluate.py` — compute obstacle_distance during eval, augment state to match model state_dim
- `scripts/merge_data.py` — pad missing state dims with 1.0 (no-obstacle) instead of 0.0
- `configs/experiment/task_b.yaml` — `state_dim: 5`, `beta_obstacle: 2.0`
- `scripts/run_task_b_5d_pipeline.sh` — full pipeline script

### Pipeline Status

1. [x] Code implementation (all 72 tests pass)
2. [ ] Data re-collection with 5D state (~15K frames, in progress)
3. [ ] Merge datasets (4D Task A padded to 5D + 5D Task B)
4. [ ] Fine-tune world model (weight surgery from 4D checkpoint, 20 epochs)
5. [ ] Refit Task B preference (K=7)
6. [ ] Evaluate Task B obstacle avoidance
7. [ ] Evaluate Task A regression

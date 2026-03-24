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

### Stage 2: Task B Data Collection (IN PROGRESS)

- Target: ~15K frames of lane-change demonstrations
- Town: Town06 (multi-lane highway)
- Agent: BehaviorAgent(aggressive) or BasicAgent fallback
- Obstacles: 3 per episode at [40, 90, 140] waypoint steps
- Speed: 20-40 km/h
- Noise tiers: clean(30%), medium(40%), high(30%)
- Output: `data/task_b_lanechange.h5`

### Stage 3: Merge Datasets (PENDING)

```bash
uv run python scripts/merge_data.py \
    data/expert_data_mixed.h5 data/task_b_lanechange.h5 \
    --output data/expert_data_v6_combined.h5
```

Expected: ~96K + ~15K = ~111K frames.

### Stage 4: Train Shared World Model (PENDING)

```bash
uv run python scripts/train.py \
    --config configs/default.yaml \
    --data data/expert_data_v6_combined.h5 \
    --output_dir outputs/train_v6_combined \
    --epochs 20
```

Fine-tuning from existing mixed-data checkpoint.

### Stage 5: Refit Task B Preference (PENDING)

```bash
uv run python scripts/refit_preference.py \
    --checkpoint outputs/train_v6_combined/checkpoints/best.pt \
    --data data/expert_data_v6_combined.h5 \
    --output outputs/train_v6_combined/checkpoints/best_taskb_pref.pt \
    --task_filter B --K 7
```

### Stage 6: Evaluate Task B (PENDING)

```bash
uv run python scripts/evaluate.py \
    --task B \
    --checkpoint outputs/train_v6_combined/checkpoints/best_taskb_pref.pt \
    --config configs/experiment/task_b.yaml \
    --num_obstacles 3 --episodes 3 --save_video \
    --output_dir outputs/eval_task_b_v1
```

## Key Parameters (Task B vs Task A)

| Parameter | Task A | Task B | Reason |
|-----------|--------|--------|--------|
| beta_state | 0.5 | 0.2 | Allow crosstrack deviation |
| horizon | 12 | 15 | Longer planning for lane-change commitment |
| noise_scale | [0.8, 0.6] | [1.0, 0.6] | Wider steer exploration |
| K (GMM) | 5 | 7 | Richer lane-change distribution |
| accel_prior | 0.3 | 0.2 | Slower obstacle approach |
| warm_start_reset_threshold | 1.0 | 4.0 | Prevent mid-lane-change reset |

## Fallback Options

- If beta_state=0.2 still prevents lane changes → enable `heading_only_state: true`
- If BehaviorAgent doesn't lane-change reliably → manual waypoint-guided collection
- If GMM collapse → increase K to 10, monitor min_component_dist

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

### Stage 4: Train Shared World Model (IN PROGRESS)

```bash
uv run python scripts/train.py \
    --config configs/default.yaml \
    --data data/expert_data_v6_combined.h5 \
    --output_dir outputs/train_v6_combined \
    --resume outputs/train_v5_finetune/checkpoints/best.pt \
    --epochs 20
```

Fine-tuning from existing mixed-data checkpoint. Epoch 1 loss: 1.5164.
Estimated completion: ~5.5 hours from start (~17 min/epoch × 20 epochs).

### Stage 5: Refit Task B Preference (PENDING — automated)

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

## Bug Fixes Applied

1. **task_b.yaml missing encoder section**: Config defaulted to `state_dim=2` instead of 4, causing `RuntimeError: size mismatch` when loading checkpoint. Fixed by adding `encoder` and `rssm` sections.
2. **Pipeline `set -e` masked by pipe**: `python ... | tee log` masks Python errors because `set -e` only checks the last pipe command (`tee`). Fixed with `set -eo pipefail`.
3. **Missing eval output directory**: `tee` failed because `outputs/eval_task_b_v1/` didn't exist. Fixed with `mkdir -p` at script start.

## Fallback Options

- If beta_state=0.2 still prevents lane changes → enable `heading_only_state: true`
- If BehaviorAgent doesn't lane-change reliably → manual waypoint-guided collection
- If GMM collapse → increase K to 10, monitor min_component_dist

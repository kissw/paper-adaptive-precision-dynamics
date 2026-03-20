# V3 Navigation Results — Bearing-Conditioned Agent

## Changes from v2
- State vector: `[speed, steer]` → `[speed, steer, bearing_to_goal]`
- Bearing: relative angle from vehicle heading to goal location (radians, -pi to pi)
- Image cropping: bottom 60% of image kept (road-focused), resized back to 64x64
- state_dim: 2 → 3 in encoder config

## Data
- 72,000 frames collected with bearing field
- Bearing distribution: mean=-0.85, std=1.66, full [-pi, pi] coverage

## Training
- 50 epochs, loss 1.54 → 1.515 (converged)
- Zero CUDA errors

## Evaluation Results

| Task | Avg Completion | Avg MLD | Notes |
|------|---------------|---------|-------|
| A (Town04) | 1.4% | 1.02m | Slight improvement from 0% |
| B (Town03) | 1.5% | 1.66m | Slight improvement from 0% |
| Baseline (Town06_Opt) | 0.5% | 4.47m | No meaningful change |

## Analysis

Adding bearing to the state provides marginal improvement (0% → 1-3% route completion) but is **insufficient for navigation**. The fundamental issue:

1. **The CEM planner optimizes EFE** which evaluates imagined trajectories against the GMM preference model
2. **The preference model is trained on ALL expert latent states** — it captures "what driving looks like" but doesn't differentiate "driving toward goal" vs "driving away from goal"
3. **Bearing is in the state** but the world model hasn't learned that bearing → steering relationship strongly enough
4. **The EFE score barely changes** with bearing because the preference GMM assigns similar probability to all driving-like states regardless of bearing

## What would actually work

### Option A: Goal-conditioned preference (principled Active Inference)
- Train separate preference models for different bearing quadrants
- Or condition the GMM on the bearing value
- The preference should assign higher probability to states with bearing → 0 (facing goal)

### Option B: Add bearing penalty to EFE (pragmatic)  
- Directly add `|bearing|` as a cost term in the EFE score
- Makes the planner prefer actions that reduce bearing to goal
- Simple but effective, doesn't require model retraining

### Option C: Hybrid controller (fastest)
- Use a PID controller for steering toward waypoints
- Use the learned model only for speed/throttle control
- Breaks pure Active Inference but gives immediate results

## Artifacts
- `data/expert_data_v3.h5` — 72k frames with bearing
- `outputs/train_v3/checkpoints/final.pt` — v3 model
- `outputs/eval_v3_nav/` — evaluation results + chase camera videos

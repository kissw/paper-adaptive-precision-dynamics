# Evaluation Results v3 — Route-Based Success Criteria

## Changes Made
1. Pre-defined routes per town with validated curves (>200m, 2+ turns)
2. Goal-reaching success criteria (distance < 10m to goal)
3. Stuck detection (speed < 0.3 m/s for 100+ frames)
4. Collision-stuck detection (collision + low speed for 60+ frames)
5. Chase camera videos (960x540 third-person behind/above vehicle)
6. Route completion percentage tracking

## Results

| Task | Town | Route | Episodes | Success | Avg Completion | Avg MLD | Termination |
|------|------|-------|----------|---------|---------------|---------|-------------|
| A | Town04 | 0 (765m) | 2 | 0/2 | 0% | 1.03m | collision_stuck, timeout |
| A | Town04 | 1 (760m) | 2 | 0/2 | 0.5% | 0.54m | collision_stuck, stuck |
| B | Town03 | 0 (417m) | 2 | 0/2 | 0% | 1.67m | timeout, timeout |
| B | Town03 | 1 (403m) | 2 | 0/2 | 0% | 1.56m | timeout, collision_stuck |
| Baseline | Town06_Opt | 0 (1332m) | 2 | 0/2 | 0% | 4.00m | collision_stuck, collision_stuck |
| Baseline | Town06_Opt | 1 (1329m) | 2 | 0/2 | 0% | 2.45m | collision_stuck, stuck |

## Analysis

### The car drives but doesn't follow routes
- The agent produces forward acceleration and steers, confirming the v2 model fix works
- However, it has no concept of route or destination — it simply drives forward using learned preferences
- Route completion stays near 0% because the car doesn't navigate toward waypoints
- The car eventually hits obstacles, goes off-road, or drives in circles

### Root cause: no navigation capability
The Active Inference agent minimizes Expected Free Energy (EFE) based on:
- **Instrumental value**: KL divergence from predicted states to preference GMM
- **Epistemic value**: ensemble disagreement for exploration

Neither of these encodes a direction or goal location. The preference model (GMM over latent states) captures "what good driving looks like" but not "where to go."

### What would be needed for route following
1. **Goal-conditioned preference**: Condition the preference model on a target waypoint direction
2. **Waypoint integration**: Feed next-waypoint bearing into the state vector
3. **Reward shaping**: Add goal-proximity reward to the EFE calculation

### Videos generated
- Chase camera (960x540) and onboard camera (64x64) videos for all episodes
- Located in `outputs/eval_v3/`

## Conclusion
The evaluation framework is now properly implemented with:
- Fixed routes with verified curves
- Goal-based success criteria
- Stuck/collision detection
- Chase camera visualization
- Route completion tracking

The agent needs navigation capability (goal conditioning) to achieve route completion.
This is a fundamental architecture enhancement, not a tuning issue.

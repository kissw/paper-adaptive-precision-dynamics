# Multi-Active Lane Change: Situation Analysis

**Date:** 2026-03-25
**Status:** Open — evasion overshoot prevents multiple active lane changes

---

## Goal

Demonstrate that the AIF obstacle avoidance agent can perform **more than one active lane-change maneuver** per episode. Currently, all obstacles are in the same lane, so after the ego changes lanes once it never encounters another obstacle.

## What Was Tried

### Attempt 1: Alternating lane obstacles (steer=±0.7, offsets=[0,1,0])

- Obstacle 0 in route lane (Y≈52), obstacle 1 in left adjacent lane (Y≈49), obstacle 2 in route lane (Y≈52)
- **Result:** 100% SR, 9/9 avoided — but only 1 active maneuver
- **Problem:** Evasion for obstacle 0 overshoots to lane -5 (Y≈45), 2 lanes from start. Obstacle 1 at Y≈49 is passively cleared with ~4m lateral margin. Obstacle 2 at Y≈52 is also passively cleared because the ego never returns to the route lane.
- Output: `outputs/eval_task_b_v5_r2_alternating/`

### Attempt 2: Two-lane offset (steer=±0.7, offsets=[0,2,0])

- Obstacle 1 placed 2 lanes left (at Y≈45, lane -5)
- **Result:** FAILED — ego overshot to lane -3 (Y≈37), went completely off-route
- **Problem:** Aggressive evasion + 2-lane offset places obstacle near road edge. Ego launched off the road entirely.

### Attempt 3: Reduced steer magnitude (steer=±0.5, offsets=[0,1,0])

- Same layout as Attempt 1 but with reduced evasion steer
- **Result:** 100% SR, 9/9 avoided — same passive clearance pattern
- **Problem:** Even with reduced steer (±0.5), CARLA vehicle dynamics produce sufficient lateral momentum to overshoot by 2 lanes. Ego lands at Y≈45 regardless of steer magnitude.
- Output: `outputs/eval_task_b_v5_r2_alternating_v2/`

## Root Cause Analysis

Three compounding factors prevent multiple active lane changes:

1. **Evasion overshoot:** The post-CEM steer override applies `evasion_steer * min(proximity*1.5, 1.0)` for multiple frames. CARLA vehicle dynamics amplify this into ~7m lateral displacement (2 lane widths), regardless of steer magnitude (0.5 or 0.7).

2. **No centering force:** Task B config uses `beta_state=0.0` (crosstrack penalty disabled) to allow lane-change deviation. This means after evasion, the ego has zero incentive to return to the original lane. The CEM planner produces near-zero steer, so the ego drifts at whatever lane it landed in.

3. **Lateral clearance threshold:** The 4m lateral clearance suppression threshold approximately equals the distance from the overshoot position (lane -5, Y≈45) to obstacles in the adjacent lane (lane -6, Y≈49). So obstacle 1 is right at the suppression boundary and never triggers active evasion.

## Potential Solutions (Within the AIF Framework)

All solutions must operate within the Active Inference framework — no classical controllers.

### A. Post-evasion centering via temporary preference shift

After passing an obstacle (lock released), temporarily increase `beta_state` or activate a centering preference that pulls the ego back toward the route lane. This is AIF-consistent: the agent's preferred state shifts from "avoid obstacle" back to "lane center" once the threat is cleared.

- **Mechanism:** When locked obstacle is cleared (`dist > min_dist + 10m`), set `beta_state=0.3` for N frames until crosstrack error < threshold, then return to `beta_state=0.0`
- **Risk:** May interfere with approach to next obstacle if spacing is tight

### B. Lane-targeted evasion (adaptive steer magnitude)

Instead of a fixed evasion steer magnitude, compute the steer needed to reach exactly the adjacent lane center (1 lane width ≈ 3.5m). Use a proportional steer that decays as the ego approaches the target lateral position.

- **Mechanism:** `evasion_steer = base_steer * max(0, 1 - lateral_progress / target_offset)`
- **AIF interpretation:** Precision of action prior decays as the ego achieves the intended lateral displacement
- **Risk:** Requires accurate lateral position tracking during the maneuver

### C. Wider obstacle spacing with more obstacles

Place 4-5 obstacles with wider spacing (fractions [0.10, 0.30, 0.50, 0.70, 0.90]) so the ego has time to drift back naturally between obstacles. With enough distance, even small CEM steer bias might bring the ego close enough to the next obstacle's lane.

- **Mechanism:** Pure route/obstacle configuration change
- **Risk:** May still not produce active maneuvers if the ego never drifts back

### D. Learned lane-return behavior

Add lane-return demonstrations to the training data. Collect episodes where the vehicle changes lanes and then returns. The preference model would then encode "return to center" as a preferred state sequence.

- **Mechanism:** New training data + preference model refit
- **AIF interpretation:** Most principled solution — the agent learns the full lane-change-and-return behavior from experience
- **Risk:** Requires significant data collection effort

## Recommendation

**Solution A** (temporary beta_state restoration) is the most practical next step. It's AIF-consistent (preference shift after threat clearance), requires minimal code changes, and directly addresses the root cause (no centering force). If it works, it also improves general driving quality between obstacles.

**Solution D** (learned lane-return) is the most principled long-term approach and should be pursued for the paper, but requires substantial data collection.

## Current State

- Route 2 definition (`lane_offsets=[0, 1, 0]`) is committed and working — it correctly places obstacles in alternating lanes
- All three routes achieve 100% SR with 28/28 obstacles avoided
- Evasion steer reverted to ±0.7 (validated for Routes 0 and 1)
- The multi-lane offset support in `obstacles.py` is functional and ready for future use

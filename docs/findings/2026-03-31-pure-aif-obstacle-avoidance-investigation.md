# 2026-03-31: Pure AIF Obstacle Avoidance Investigation

## Goal

Redesign Task B (obstacle avoidance) to use pure Active Inference — no privileged
ground-truth obstacle coordinates. Obstacle detection must come from visual input,
lane change must be triggered by EFE minimization, and the system must stay within
the AIF framework.

## Approach

Removed 6 privileged mechanisms (runtime coordinates, evasion direction, reflexive
steer override, motor primitives, directional CEM penalty, obstacle EFE penalty)
and replaced with pure EFE scoring: instrumental (GMM) + epistemic (ensemble) +
state-space (heading + crosstrack).

The hypothesis: a preference model fitted on obstacle-free expert driving assigns
low probability to obstacle-present latent states, making the EFE naturally high
for "stay in lane" trajectories and low for "change lane" trajectories.

## Experiments

### 1. Preference Gap Analysis (Single GMM)

| Model | GMM Gap (nats) | Recon Error Ratio | Avoidance |
|-------|---------------|-------------------|-----------|
| 64×64 RGB | -24.58 | 1.62x | 0/9 (0%) |
| 128×128 RGB | -2.29 | 1.91x | 0/9 (0%) |
| 64×64 RGBD | -68.05 | 1.35x | not eval'd |

**Finding:** The GMM gap is *inverted* in all cases — obstacle-present latents
get higher probability than obstacle-free latents. The world model's latent space
encodes driving dynamics (speed, steer, lane position), not scene content (obstacle
presence).

### 2. Beta Instrumental Tuning

| β_instrumental | Lane Changes | Avoidance | Effect |
|---------------|-------------|-----------|--------|
| 1.0 (default) | 9/ep | 0% | baseline |
| 3.0 | 3/ep | 0% | worse — amplifies straight-driving preference |

### 3. Visual Surprise (Reconstruction Error)

Added reconstruction error as EFE penalty: obstacle frames have 1.62x higher
reconstruction error than clean frames. When surprise exceeds threshold, penalize
imagined trajectories whose decoded images stay similar to current observation.

**Result:** 0% avoidance. More lane changes (12 vs 9) but not timed to obstacles.
RSSM imagination doesn't predict obstacle appearance/disappearance accurately.

### 4. Contrastive Preference (Log-Ratio of Two GMMs)

Replaced single GMM with `log p_clean(z) - scale * log p_avoid(z)`:

| Scale | Effective Gap | MLD | Avoidance | Driving Quality |
|-------|-------------|------|-----------|----------------|
| 1.0 | 37.02 nats | 4.0m | 0% | Broken (off-road) |
| 0.3 | 9.24 nats | 0.32m | 0% | Degraded (14-31% compl) |
| 0.1 | 0.99 nats | 0.31m | 0% | Good but no avoidance |

**Finding:** The contrastive preference correctly separates latents (37 nats gap)
but RSSM imagination is obstacle-blind. CEM trajectories produce similar latents
regardless of steering direction because the world model cannot predict obstacle
dynamics during open-loop imagination.

### 5. 128×128 Resolution

| Metric | 64×64 | 128×128 |
|--------|-------|---------|
| Success Rate | 0% | 100% |
| MLD | 0.31m | 0.17m |
| Offroad | yes | 0 |
| Avoidance | 0% | 0% |

**Finding:** Higher resolution dramatically improves driving quality but does not
help obstacle avoidance. The 128×128 model is the best pure-AIF driver we've
produced (100% SR, 0.17m MLD, 0 offroad events).

## Root Cause

The bottleneck is **not** the preference model — it's the **world model's imagination**.

During CEM planning, the RSSM's `imagine()` function rolls out trajectories using
`img_step` (prior prediction without observation). These imagined latent sequences
do not encode obstacle proximity because:

1. **Insufficient obstacle training data:** Only 30K obstacle frames (24% of 126K),
   all with collisions (BehaviorAgent had 0% collision-free success rate). The model
   never sees the complete "approach → lane change → pass → clear" visual sequence.

2. **Encoder compression:** At 64×64 with 4 stride-2 conv layers (→ 4×4 spatial),
   obstacles at highway distance are compressed away. Even at 128×128 with 5 layers
   (→ 4×4), the bottleneck loses obstacle detail.

3. **Latent space organization:** The RSSM latent encodes driving dynamics (temporal
   autocorrelation of speed, steer, heading) rather than scene content. Obstacle
   presence doesn't create distinct latent clusters.

## AIF Interpretation

In Active Inference, obstacle avoidance through EFE minimization requires the
generative model to predict the **sensory consequences of actions** — specifically,
that steering left/right causes the obstacle to exit the visual field. Without this
predictive capability, the agent cannot evaluate whether a lane-change trajectory
leads to a preferred (obstacle-free) state.

The preference model is solved (contrastive gives 37 nats separation at the
filtering level). But the planning loop operates through imagination, and the
imagined trajectories are obstacle-blind. This is analogous to an agent that can
recognize danger when it sees it but cannot predict danger from its plans.

## Conclusion

Pure vision-based AIF obstacle avoidance via EFE minimization is not viable with
the current RSSM architecture and training data. The contrastive preference
approach correctly identifies obstacle-present vs obstacle-free states in the
posterior (filtering), but the prior (imagination) cannot propagate this information
through action-conditioned rollouts.

## Next Steps

1. **Clean obstacle avoidance demonstrations** — fix data collection to produce
   collision-free lane-change episodes, giving the world model the complete visual
   sequence of successful obstacle avoidance
2. **Auxiliary obstacle prediction loss** — supervised head during training that
   forces the latent space to encode obstacle presence
3. **Object-centric world models** — architectures (SLATE, SAVi) that naturally
   separate objects from background in the latent representation

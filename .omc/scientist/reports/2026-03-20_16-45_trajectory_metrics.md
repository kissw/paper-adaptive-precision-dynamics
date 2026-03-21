# Trajectory Metrics Report
**Generated:** 2026-03-20 16:45 UTC
**Analysis:** Active Inference Agent — Evaluation Trajectories

---

## Objectives

[OBJECTIVE] Compute paper-quality driving performance metrics from CARLA evaluation trajectory JSONL files for two conditions:
- **Baseline** (Town06_Opt): 6 episodes (r0_ep0–2, r1_ep0–2); r0_ep0 is truncated (638 frames; 5 complete at 2,000 frames each)
- **Task A** (Town04): 6 episodes (r0_ep0–2, r1_ep0–2); all 6,000 frames each

---

## Data Summary

[DATA]
| Condition | Episodes | Frames/Episode | Total Frames | Timestep (s) |
|-----------|----------|---------------|-------------|--------------|
| Baseline (Town06_Opt) | 6 (5 complete + 1 truncated) | 2,000 (complete) / 638 (r0_ep0) | 10,638 | 0.05 |
| Task A (Town04) | 6 | 6,000 | 36,000 | 0.05 |

Fields per frame: frame_id, timestamp, x, y, z, yaw, speed_mps, steer, action_steer, action_accel, lateral_dev, lane_id, road_id, collision, lane_invasion, efe_score, epistemic_score, route_completion_pct, distance_to_goal

---

## Per-Episode Breakdown

### Baseline (Town06_Opt)

| Episode | Frames | Dist (m) | Speed mean±std (m/s) | Lat Dev mean±std (m) | Lat Dev max (m) | Collisions | Lane Inv | EFE mean | Epistemic mean | Route % max |
|---------|--------|----------|---------------------|---------------------|-----------------|------------|----------|----------|---------------|------------|
| r0_ep0* | 638 | 140.7 | 4.43 ± 1.31 | 0.027 ± 0.028 | 0.128 | 0 | 0 | — | — | 10.7% |
| r0_ep1 | 2,000 | 488.7 | 4.89 ± 0.81 | 0.012 ± 0.027 | 0.173 | 0 | 0 | 16.891 | 0.457 | 27.2% |
| r0_ep2 | 2,000 | 488.8 | 4.89 ± 0.80 | 0.012 ± 0.027 | 0.168 | 0 | 0 | 16.893 | 0.457 | 27.2% |
| r1_ep0 | 2,000 | 489.6 | 4.90 ± 0.79 | 0.012 ± 0.027 | 0.168 | 0 | 0 | 16.884 | 0.457 | 27.3% |
| r1_ep1 | 2,000 | 489.2 | 4.90 ± 0.79 | 0.012 ± 0.028 | 0.171 | 0 | 0 | 16.888 | 0.457 | 27.3% |
| r1_ep2 | 2,000 | 489.3 | 4.90 ± 0.78 | 0.012 ± 0.027 | 0.169 | 0 | 0 | 16.887 | 0.457 | 27.3% |

*r0_ep0 is a truncated/incomplete run (638 frames vs expected 6,000); excluded from aggregate statistics below.

### Task A (Town04)

| Episode | Frames | Dist (m) | Speed mean±std (m/s) | Lat Dev mean±std (m) | Lat Dev max (m) | Collisions | Lane Inv | EFE mean | Epistemic mean | Route % max |
|---------|--------|----------|---------------------|---------------------|-----------------|------------|----------|----------|---------------|------------|
| r0_ep0 | 6,000 | 1508.2 | 5.03 ± 0.48 | 0.110 ± 0.150 | 0.556 | 0 | 0 | 16.825 | 0.463 | 100.0% |
| r0_ep1 | 6,000 | 1508.6 | 5.03 ± 0.47 | 0.110 ± 0.150 | 0.566 | 0 | 0 | 16.825 | 0.464 | 100.0% |
| r0_ep2 | 6,000 | 1510.1 | 5.04 ± 0.47 | 0.110 ± 0.150 | 0.579 | 0 | 0 | 16.825 | 0.464 | 100.0% |
| r1_ep0 | 6,000 | 1509.1 | 5.03 ± 0.47 | 0.110 ± 0.151 | 0.568 | 0 | 0 | 16.825 | 0.464 | 100.0% |
| r1_ep1 | 6,000 | 1509.6 | 5.03 ± 0.47 | 0.110 ± 0.150 | 0.571 | 0 | 0 | 16.825 | 0.464 | 100.0% |
| r1_ep2 | 6,000 | 1508.9 | 5.03 ± 0.47 | 0.110 ± 0.150 | 0.576 | 0 | 0 | 16.825 | 0.463 | 100.0% |

---

## Aggregate Metrics (Paper Table)

[FINDING] Task A (Town04) achieved 100% route completion across all 6 episodes while Baseline (Town06_Opt) reached only 27.3% in complete episodes — indicating the baseline evaluation was frame-budget limited (2,000 frames) rather than a route completion failure.
[STAT:n] Baseline: n=10,000 frames (5 complete eps); Task A: n=36,000 frames (6 eps)

| Metric | Baseline (Town06_Opt) | Task A (Town04) |
|--------|-----------------------|-----------------|
| Episodes (complete) | 5 | 6 |
| Frames per episode | 2,000 ± 0 | 6,000 ± 0 |
| **Total distance (m)** | **489.1 ± 0.3** | **1,509.1 ± 0.6** |
| **Mean speed (m/s)** | **4.896 ± 0.003** | **5.033 ± 0.002** |
| Speed std (m/s) | 0.794 ± 0.009 | 0.473 ± 0.003 |
| **Mean lat dev (m)** | **0.0120 ± 0.0001** | **0.1097 ± 0.0002** |
| Lat dev std (m) | 0.0272 ± 0.0003 | 0.1504 ± 0.0002 |
| **Max lat dev (m)** | **0.169 ± 0.002** | **0.569 ± 0.014** |
| **Collisions** | **0** | **0** |
| **Lane invasions** | **0** | **0** |
| **Mean EFE score** | **16.889 ± 0.004** | **16.825 ± 0.002** |
| EFE std | 0.247 ± 0.003 | 0.181 ± 0.002 |
| **Mean epistemic score** | **0.4568 ± 0.0002** | **0.4635 ± 0.0002** |
| Epistemic std | 0.0193 ± 0.0003 | 0.0179 ± 0.0001 |
| **Route completion max (%)** | **27.3 ± 0.0** | **100.0 ± 0.0** |

---

## Statistical Comparisons

### Lateral Deviation (Baseline vs Task A)

[FINDING] Task A exhibits significantly higher lateral deviation than Baseline. This likely reflects the more complex road geometry of Town04 (curves, intersections) compared to the predominantly straight Town06_Opt route.
[STAT:effect_size] Cohen's d = 0.905 (large effect)
[STAT:p_value] Welch t-test: t = -116.65, df = 42,836, p << 0.001
[STAT:n] Baseline n = 10,000 frames; Task A n = 36,000 frames
[STAT:ci] Baseline mean lat dev = 0.012 m; Task A mean lat dev = 0.110 m (difference = 0.098 m)

### Speed (Baseline vs Task A)

[FINDING] Task A mean speed is marginally higher than Baseline (5.033 vs 4.896 m/s), with lower speed variance (std 0.473 vs 0.794 m/s), suggesting smoother cruise control on Town04's longer straight sections.
[STAT:effect_size] Cohen's d = 0.209 (small effect)
[STAT:p_value] Welch t-test: t = -16.45, df = 12,037, p << 0.001
[STAT:n] Baseline n = 10,000; Task A n = 36,000

### Safety Metrics

[FINDING] Both conditions achieved zero collisions and zero lane invasions across all episodes and all frames recorded.
[STAT:n] Baseline: 10,638 total frames; Task A: 36,000 total frames

---

## Trajectory Shape Analysis

[FINDING] The x,y coordinate progressions are consistent with vehicles driving on structured road networks, not random motion. Baseline follows a predominantly rectilinear corridor (~478 m east-west with minor lateral variation ~40 m), while Task A traces a closed circuit loop with turns (bounding box ~368 x 247 m).

| Condition | X range (m) | Y range (m) | Bounding Box Area (m²) | Route Type |
|-----------|------------|------------|----------------------|------------|
| Baseline (Town06_Opt) | 183–661 (~478 m) | 47–87 (~40 m) | ~18,900–19,200 | Straight corridor (highway-like) |
| Task A (Town04) | 14–383 (~368 m) | -365 to -118 (~247 m) | ~91,040–91,048 | Closed circuit loop |

Key observations:
- **Baseline**: Yaw stays near 0° on the main stretch; the vehicle accelerates from ~183 m east, travels ~478 m, consistent with a straight highway segment. Route caps at 27.3% → frame budget (2,000 at 0.05s = 100s) exhausted before full route completion.
- **Task A**: Yaw rotates through full 360° range (0°, 90°, -180°, -74°, 91°, 180°) confirming multi-turn circuit. Episodes complete 100% route, then the agent continues lapping (route_completion_pct resets to ~0.3% at frame 3000 in r0_ep0).

---

## Limitations

[LIMITATION] Baseline episodes are frame-budget limited (2,000 frames = 100s), not route-completion limited. The truncated r0_ep0 (638 frames) is excluded from aggregates. Re-run with 6,000 frames is in progress; current results underrepresent baseline route coverage.

[LIMITATION] Frame-level collision and lane_invasion flags are binary per-frame counts; consecutive stuck frames (e.g., after a collision) would inflate collision counts — though both tasks show zero here.

[LIMITATION] Lateral deviation is computed relative to the lane center as reported by CARLA's waypoint system; measurement accuracy depends on the map's road topology representation.

[LIMITATION] Statistical tests pool frames across episodes, violating independence assumptions (frames within an episode are temporally correlated). The reported p-values are indicative only; episode-level (n=5–6) is the true independent sample size for inferential purposes.

[LIMITATION] Task A's higher lateral deviation may partly reflect the longer frame window (6,000 vs 2,000) capturing more of the circuit where lateral errors accumulate on curves.

[LIMITATION] EFE and epistemic scores are nearly identical across conditions (< 0.07 difference in means), suggesting the active inference objective function is robust to environment changes at this level of analysis.

---

## Figures

- `fig1_trajectories.png` — X/Y trajectory plots for both conditions
- `fig2_timeseries.png` — Speed, lateral deviation, EFE, epistemic over time
- `fig3_per_episode_bars.png` — Per-episode metric bar charts
- `fig4_latdev_hist.png` — Lateral deviation distributions

All figures saved to: `.omc/scientist/figures/`

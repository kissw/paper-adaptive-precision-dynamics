# Fix Data Collection, Training, and Evaluation Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the three root causes preventing the Deep Active Inference agent from driving: (1) runaway speed in data collection, (2) braking-biased actions, (3) undertrained RSSM action-conditioning. Then recollect data, retrain, and evaluate.

**Architecture:** Fix OU noise to respect autopilot intent (preserve expert action sign when overspeed). Increase RSSM kl_rep_scale so the prior pathway learns action-conditioned dynamics. Keep the already-applied EFE fixes (cross-entropy normalization, min_std=1.0, beta_epistemic=1.0). Recollect ~72k frames, retrain 50 epochs, evaluate on CARLA.

**Tech Stack:** Python 3.12, PyTorch 2.10, CARLA 0.9.16, uv, HDF5

**Evidence from debugging:**
- Empirical test: BasicAgent at 30 km/h → get_velocity()=8.1 m/s. Training data mean=123 m/s (15x too high)
- OU noise overrides autopilot braking on straight highway → speed runaway
- Expert actions: 96.3% brake (accel<0), 3.7% throttle → model learns "driving=braking"
- RSSM imagination: FORWARD vs ZERO action plans differ by <0.01 in preference log_prob → CEM planner blind
- Preference GMM: old min_std=0.01 → KL=56,000 for all trajectories (already fixed to min_std=1.0)

---

### Task 1: Fix OU Noise — Prevent Speed Runaway

**Problem:** OU noise can flip the sign of the expert action's accel component. When the autopilot says "brake hard" (-0.8) but noise adds +0.5, the applied action is -0.3 (weak brake). On long straights, this causes speed to accumulate to 245 m/s.

**Files:**
- Modify: `scripts/collect_data.py` (lines 128-132)
- Test: manual verification during data collection

**Step 1: Modify noise application to preserve expert accel sign when overspeed**

In `scripts/collect_data.py`, replace the noise application block (lines 128-132):

```python
# CURRENT (broken):
if tier == "random" and rng.random() < 0.3:
    noisy_action = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
else:
    noisy_action = np.clip(expert_action + noise.sample(), -1.0, 1.0)
```

Replace with:

```python
if tier == "random" and rng.random() < 0.3:
    noisy_action = np.array([rng.uniform(-1, 1), rng.uniform(-0.3, 0.5)])
else:
    noise_sample = noise.sample()
    noisy_action = np.clip(expert_action + noise_sample, -1.0, 1.0)
    # Prevent noise from overriding braking when overspeed
    speed_kmh = state[0] * 3.6  # state[0] is m/s from get_velocity()
    if speed_kmh > target_speed * 1.1 and expert_action[1] < 0:
        noisy_action[1] = min(noisy_action[1], expert_action[1])
```

This ensures: when the car exceeds 110% of target speed AND the autopilot is braking, noise cannot weaken the brake. Random tier accel is also bounded to [-0.3, 0.5] instead of full [-1, 1] to prevent sustained runaway.

**Step 2: Also limit random tier accel**

The random tier currently generates `rng.uniform(-1, 1)` for accel. A sustained +1.0 throttle from random episodes will cause runaway. Cap it.

Already handled in step 1 above: `rng.uniform(-0.3, 0.5)`.

---

### Task 2: Fix Speed Recording — Store km/h to Match Agent Internals

**Problem:** `carla_env.py` stores speed in m/s, but the model encoder sees this as a feature. During evaluation, the car starts at 0 m/s which is in-distribution, but during training the speeds ranged 0-245 m/s. After fixing the runaway, speeds will be 0-14 m/s (0-50 km/h). Either way is fine as long as training and evaluation are consistent. No change needed here — just documenting.

**Files:** No changes needed. The m/s recording is correct. After fixing the noise, speeds will naturally be in the 0-14 m/s range, matching evaluation.

---

### Task 3: Fix RSSM Training — Strengthen Action-Conditioning

**Problem:** `kl_rep_scale=0.1` means the posterior is barely pushed toward the prior. The prior pathway (which uses actions as input to the GRU) gets weak gradients. Result: the RSSM's `imagine()` produces nearly identical trajectories regardless of action input.

**Files:**
- Modify: `configs/default.yaml` (line 24)

**Step 1: Increase kl_rep_scale**

```yaml
# CURRENT:
kl_rep_scale: 0.1

# CHANGE TO:
kl_rep_scale: 0.5
```

Rationale: DreamerV3 uses equal scales (0.5/0.5). Our 0.1 was too asymmetric — the prior never had to accurately predict next states conditioned on actions. At 0.5, the prior must match the posterior more closely, forcing it to actually use the action input.

---

### Task 4: Verify EFE Fixes Are in Place

**Problem:** Already fixed in previous session, but verify the changes persist.

**Files:**
- Verify: `configs/default.yaml` — `preference.min_std: 1.0`, `efe.beta_epistemic: 1.0`
- Verify: `src/active_inference/planning/efe.py` — uses cross-entropy/dim instead of raw KL

**Step 1: Confirm config**

```yaml
preference:
  min_std: 1.0      # was 0.01 (collapsed GMM)

efe:
  beta_epistemic: 1.0  # was 0.1 (drowned by instrumental)
```

**Step 2: Confirm EFE scorer uses normalized cross-entropy**

`instrumental_value()` should compute `-log_p.mean(0) / latent_dim` instead of `(log_q - log_p).mean(0)`.

---

### Task 5: Recollect Data

**Prerequisites:** CARLA 0.9.16 server running, fixes from Task 1 applied.

**Files:**
- Run: `scripts/collect_data.py`
- Output: `data/expert_data_v2.h5`

**Step 1: Start CARLA server**

```bash
nohup /data/jaerock/carla-0.9.16/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 > /tmp/carla_server.log 2>&1 &
```

**Step 2: Run data collection**

```bash
export PYTHONPATH="/data/jaerock/carla-0.9.16/PythonAPI/carla:$PYTHONPATH"
uv run python scripts/collect_data.py \
  --town Town06_Opt \
  --num_samples 72000 \
  --output data/expert_data_v2.h5 \
  --host 127.0.0.1 \
  --port 2000 \
  --min_speed 15 \
  --max_speed 50
```

**Step 3: Verify data quality**

```bash
uv run python -c "
import h5py, numpy as np
with h5py.File('data/expert_data_v2.h5', 'r') as f:
    s = f['states'][:]
    a = f['actions'][:]
    print(f'Speed (m/s): mean={s[:,0].mean():.2f}, max={s[:,0].max():.2f}')
    print(f'Expected range: 0-14 m/s (0-50 km/h)')
    print(f'Accel>0 (throttle): {(a[:,1]>0).mean():.1%}')
    print(f'Expected: >30% throttle')
"
```

**Expected:** Speed mean ~8 m/s (30 km/h), max <16 m/s (58 km/h). Throttle fraction >30%.

---

### Task 6: Retrain from Scratch

**Prerequisites:** `data/expert_data_v2.h5` exists with corrected data.

**Files:**
- Run: `scripts/train.py`
- Config: `configs/default.yaml` (already updated with kl_rep_scale=0.5, preference.min_std=1.0, efe.beta_epistemic=1.0)

**Step 1: Launch training**

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/train.py \
  --config configs/default.yaml \
  --data data/expert_data_v2.h5 \
  --output_dir outputs/train_v2 \
  --epochs 50 \
  > outputs/train_v2/training.log 2>&1 &
```

**Step 2: Monitor**

```bash
tail -f outputs/train_v2/training.log
```

**Expected:** ~7 hours. Loss should converge. GMM updates from epoch 10.

---

### Task 7: Verify RSSM Action-Conditioning

**Prerequisites:** Training complete (outputs/train_v2/checkpoints/final.pt exists).

**Step 1: Test imagination divergence**

Run the same diagnostic script from the debugging session. FORWARD vs ZERO trajectories should show meaningful log_prob difference (>2.0) under the preference model, and EFE scores should clearly differentiate FORWARD from ZERO (FORWARD should score BETTER, i.e., lower EFE).

**Pass criteria:**
- FORWARD EFE < ZERO EFE (planner prefers driving forward)
- Imagined trajectory log_prob difference > 1.0 between FORWARD and ZERO
- CEM planner outputs consistent positive accel over 10 steps

If this check FAILS, the kl_rep_scale increase was insufficient. Next step would be adding an action prediction auxiliary loss to the training (but try 0.5 first).

---

### Task 8: Evaluate in CARLA

**Prerequisites:** Task 7 passes, CARLA server running.

**Step 1: Run evaluation**

```bash
export PYTHONPATH="/data/jaerock/carla-0.9.16/PythonAPI/carla:$PYTHONPATH"

# Task A — Town04 lane keeping
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py \
  --task A --checkpoint outputs/train_v2/checkpoints/final.pt \
  --save_video --output_dir outputs/eval_v2

# Task B — Town03 obstacles
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py \
  --task B --checkpoint outputs/train_v2/checkpoints/final.pt \
  --save_video --output_dir outputs/eval_v2

# Baseline — Town06
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py \
  --task baseline --checkpoint outputs/train_v2/checkpoints/final.pt \
  --save_video --output_dir outputs/eval_v2
```

**Step 2: Verify the car actually moves**

Check eval videos manually. The car should:
- Move forward at sustained speed
- Stay in lane (Task A: MLD < 0.5m)
- Avoid obstacles (Task B)
- Complete 1000 frames per episode

**Pass criteria:**
- Success rate > 60% on Task A
- Car visibly drives (not stationary)
- Mean lateral deviation < 1.0m

---

### Task 9: Analyze Results

**Step 1: Read CSV**

```bash
cat outputs/eval_v2/eval_results.csv
```

**Step 2: Compare with expected baselines**

| Metric | Task A Target | Task B Target | Notes |
|---|---|---|---|
| Success Rate | >60% | >40% | First model, conservative targets |
| Mean Lateral Dev | <1.0m | <2.0m | Obstacle avoidance increases MLD |
| Off-Road Events | <5 total | <10 total | |
| Frames/Episode | 1000 | >500 | Collision ends episode early |

If targets not met but car moves: the Active Inference pipeline works, just needs tuning.
If car still doesn't move: escalate to adding action prediction auxiliary loss.
</content>
</invoke>
# System Design — Deep Active Inference Agent

Technical architecture reference for the Deep Active Inference autonomous driving agent.

---

## 1. System Overview

The agent loop runs at each simulation step:

```
Observe → Encode → RSSM Posterior → iCEM Plan (EFE Scoring) → Act
```

```
                    ┌─────────────────────────────────────────────┐
                    │              DeepAIFAgent                    │
                    │                                             │
  image (64×64×3) ──┤  ┌────────────┐   ┌──────┐                 │
  state (4D)  ──────┤  │ ConvEncoder │──▶│ RSSM │──▶ posterior    │
                    │  └────────────┘   └──┬───┘    (z_t)        │
                    │                      │                      │
                    │              ┌───────▼────────┐             │
                    │              │  iCEM Planner   │             │
                    │              │  (500 samples)  │             │
                    │              │                 │             │
                    │              │  for each candidate action:  │
                    │              │    RSSM.imagine ──▶ EFEScorer │
                    │              │    select elites, refit      │
                    │              └───────┬────────┘             │
                    │                      │                      │
                    │              action [steer, accel]          │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           ▼
                                    CARLA Simulator
```

---

## 2. Component Architecture

### WorldModel

Container for all learned components. Defined in `src/active_inference/agent.py`.

```python
class WorldModel(nn.Module):
    encoder: ConvEncoder       # image + state → 256D embedding
    rssm: RSSM                 # sequence model (deter=256, stoch=64)
    obs_decoder: ObsDecoder    # features → 64×64 image reconstruction
    state_decoder: StateDecoder # features → 4D state reconstruction
    ensemble: EnsembleTransitionHeads  # 5 MLP heads for epistemic uncertainty
```

### RSSM (Recurrent State-Space Model)

File: `src/active_inference/models/rssm.py`

State representation: `RSSMState(mean, std, stoch, deter)` where features = `cat(deter, stoch)` → 320D.

Three forward modes:
- **`obs_step(prev_state, prev_action, embed)`** — posterior update with observation (training + online inference)
- **`img_step(prev_state, prev_action)`** — prior-only prediction (imagination during planning)
- **`imagine(initial_state, actions)`** — multi-step open-loop rollout for CEM trajectory scoring

GRU dynamics: `deter_t = GRU(cat(stoch_{t-1}, action_{t-1}), deter_{t-1})`
Stochastic: `stoch_t ~ N(mean_t, std_t)` where `[mean, std] = MLP(deter_t [+ embed_t])`

### ConvEncoder

File: `src/active_inference/models/encoder.py`

Dual-input encoder fusing visual and proprioceptive information:
- **Image path:** 64×64×3 RGB → 4-layer CNN (channels: 32→64→128→256) → 1024D
- **State path:** 4D state → MLP(4→64→128) → 128D
- **Fusion:** `cat(image_feat, state_feat)` → MLP → 256D embedding

Optional `crop_road=True` crops top 25% of image (sky removal) before encoding.

### ObsDecoder

File: `src/active_inference/models/decoder.py`

320D features → ConvTranspose2d (4 layers, reverse of encoder) → 64×64×3 reconstructed image.

### StateDecoder

File: `src/active_inference/models/decoder.py`

320D features → MLP(320→256→4) → reconstructed state `[speed, steer, heading_error, crosstrack_error]`.

Used during planning for state-space EFE penalty (heading and crosstrack error minimization).

### EnsembleTransitionHeads

File: `src/active_inference/models/ensemble.py`

5 independent MLP heads, each predicting stochastic state parameters from features. Disagreement between heads estimates epistemic uncertainty (information gain) for the epistemic term in EFE.

### PreferenceModel (GMM)

File: `src/active_inference/training/preference.py`

Gaussian Mixture Model in 64D stochastic latent space:
- Task A: K=5 components, fit on success-only expert latents
- Task B: K=7 components, fit on lane-change episode latents

Trained via MLE (`fit_iters=300`, `fit_lr=0.005`) on latent samples from the encoder. Updated periodically during world model training.

---

## 3. EFE Formulation

File: `src/active_inference/planning/efe.py`

```
EFE(τ) = Σ_t γ^t [ β_i · E_q[-log p_pref(z_t)]        (instrumental)
                   + β_e · H_ensemble(z_t)                (epistemic)
                   + β_s · (h_t² + c_t²)                  (state penalty)
                   + β_o · obstacle_penalty(s_t) ]         (obstacle, Task B only)
```

### Instrumental Value
Cross-entropy `E_q[-log p_pref(z)]` between predicted latent distribution and GMM preference. MC-sampled (32 samples). Measures divergence from preferred latent states.

### Epistemic Value
Entropy of predicted stochastic state estimated via ensemble disagreement. Drives exploration toward informative states.

### State Penalty
Decoded state-space deviation: `heading_error² + crosstrack_error²`, clamped at 4.0. Uses raw values, NOT z-scored — state decoder predictions during open-loop imagination are noisy. Task B uses heading-only variant (no crosstrack penalty during lane changes).

### Obstacle Proximity Penalty (Task B)
Runtime penalty based on forward obstacle detection. Not part of learned model — computed from CARLA actor positions during evaluation.

### Temporal Discount
`γ = 0.95` — later timesteps contribute less to total EFE score.

### Z-Score Normalization
Instrumental and epistemic terms are z-scored per-timestep within each CEM batch to ensure comparable scales. State penalty is NOT z-scored.

---

## 4. Training Pipeline

File: `src/active_inference/training/losses.py`

### Variational Free Energy (VFE) Loss

```
VFE = image_recon_loss + state_recon_loss + kl_loss
```

- **Image reconstruction:** MSE between decoded and target images
- **State reconstruction:** MSE with symlog transform on target states (compresses large errors)
- **KL divergence:** DreamerV3-style dual KL: `kl_dyn * KL(sg(posterior) || prior) + kl_rep * KL(posterior || sg(prior))` with free nats (1.0) threshold

### Training Loop
- Per-timestep backward with AMP (automatic mixed precision)
- NaN guard: skip batch on NaN loss, CUDA recovery on device errors
- Gradient clipping: max norm 100.0
- Learning rate: 1e-4 with warmup (1000 steps)
- Preference model updated every epoch after warmup (epoch 10)

---

## 5. Planning Pipeline

File: `src/active_inference/planning/cem_planner.py`

### iCEM Loop (per step)

1. **Initialize:** Mean from warm-start (shifted previous solution) or zero. Std from config.
2. **Sample:** Generate N=500 action sequences using colored noise (1/f^β, β=1.0) for temporal smoothness
3. **Action priors:** Initialize accel channel to 0.3 (breaks cold-start symmetry)
4. **Imagine:** RSSM rollout for each candidate → predicted latent trajectories
5. **Score:** EFEScorer evaluates each trajectory
6. **Select elites:** Top 50 trajectories by EFE score
7. **Refit:** Update mean and std from elite statistics
8. **Repeat:** 5 iterations (10 on cold-start)
9. **Return:** First action from best trajectory

### Cold-Start Recovery
When no warm-start available (first step or after reset):
- Double sample count (1000 vs 500)
- Extra iterations (10 vs 5)
- Wider initial std

### Warm-Start
Shift previous solution forward by 1 timestep, append zero action at end. Reset if trajectory EFE exceeds threshold.

### Action Priors and Motor Primitives (Task B)
During obstacle evasion, a high-precision action prior overrides CEM steer output. Implemented in `scripts/evaluate.py`:
- Proximity-scaled steer override: `action[0] = evasion_steer * min(proximity*1.5, 1.0)`
- Lock persistence ensures committed lane change
- Lateral clearance >= 4m suppresses evasion (already in safe lane)

---

## 6. Data Flow

```
CARLA Simulator
      │
      ▼
  Data Collection (scripts/collect_data.py)
      │  images [N,3,64,64], states [N,4], actions [N,2],
      │  episode_ids, success_flags, task_labels
      ▼
  HDF5 Dataset (data/*.h5)
      │
      ├──▶ Merge (scripts/merge_data.py) ──▶ Combined HDF5
      │
      ▼
  Training (scripts/train.py)
      │  VFE minimization: image + state + KL losses
      │  Preference GMM fitting on success latents
      ▼
  Checkpoint (outputs/train_*/checkpoints/best.pt)
      │
      ├──▶ Refit Preference (scripts/refit_preference.py)
      │    Task-specific GMM (K, data filter)
      │
      ▼
  Evaluation (scripts/evaluate.py)
      │  Online inference: observe → encode → plan → act
      │  Metrics: completion, MLD, obstacle avoidance
      ▼
  Results (outputs/eval_task_*/)
      trajectories, videos, metrics
```

---

## 7. Action Space

Two-dimensional continuous: `[steer, accel]`, both in [-1, 1].

| Channel | Range | CARLA Mapping |
|---------|-------|---------------|
| steer | [-1, 1] | Direct steering angle |
| accel | [-1, 1] | Linear map to throttle [0.35, 0.55] |

No braking — accel channel controls speed within a narrow throttle band. Negative accel values still map to positive throttle (minimum 0.35).

---

## 8. Configuration System

File: `src/active_inference/config.py`

Dataclass-based with YAML overlay via OmegaConf:

```python
@dataclass
class Config:
    rssm: RSSMConfig          # deter_dim, stoch_dim, embed_dim, min_std
    encoder: EncoderConfig    # image_channels, image_size, state_dim, crop_road
    training: TrainingConfig  # lr, epochs, batch_size, seq_len, grad_clip, KL params
    ensemble: EnsembleConfig  # num_heads, hidden_dim
    cem: CEMConfig            # horizon, n_samples, n_elites, noise params
    preference: PreferenceConfig  # K, min_std, update_interval
    efe: EFEConfig            # beta_instrumental, beta_epistemic, beta_state, beta_obstacle
    evaluation: EvaluationConfig  # episodes, max_frames, towns
    data: DataConfig          # CARLA version, collection params
```

`Config.from_yaml(path)` merges schema defaults with YAML overrides.

### Config Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Production config (Task A defaults) |
| `configs/experiment/debug.yaml` | Tiny model for fast unit tests |
| `configs/experiment/task_b.yaml` | Task B overrides (initial, deprecated) |
| `configs/experiment/task_b_v5.yaml` | Task B v5 obstacle avoidance (active) |

---

## 9. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Route Completion (%) | Distance traveled / total route distance |
| MLD (m) | Mean Lateral Deviation from lane center |
| Success Rate (SR) | Episodes reaching goal / total episodes |
| Obstacle Avoidance | Obstacles passed safely / total obstacles |
| Lane Changes | Number of detected lane crossings |
| Offroad Events | Frames where vehicle leaves drivable surface |

---

## 10. Key Design Decisions

### State Penalty: Raw Values, Not Z-Scored
State decoder predictions during open-loop imagination are noisy. Z-scoring amplifies this noise and causes agent crashes (121 frames vs 925 without). Raw values with `clamp(max=4.0)` provide a gentle directional nudge. `beta_state=0.3–0.5` works; `beta_state=1.0` with z-scoring crashes.

### Cold-Start Recovery
First planning step has no warm-start trajectory. Doubling samples (500→1000) and iterations (5→10) ensures the planner finds a reasonable initial policy. Without this, ~50% of episodes fail immediately.

### Accel Prior
CEM initializes the accel channel mean to 0.3 (not 0.0). This breaks the symmetry where the planner might explore braking-like actions at the start, giving the vehicle forward momentum for the state decoder to work with.

### Colored Noise
1/f^β noise (β=1.0) for temporally smooth action sequences. White noise produces jerky, unrealistic trajectories that score poorly.

### Dual-Space EFE
Neither latent GMM preference alone (too coarse for fine lane centering) nor state penalty alone (too noisy) is sufficient. The combination — GMM for general trajectory quality + state penalty for precise navigation — produces stable lane keeping.

### Runtime Obstacle Penalty (Not Learned)
Retraining the world model with obstacle state (5D) via weight surgery destroyed learned steering (81% → 2.6% completion). The runtime approach keeps the proven 4D model and adds obstacle awareness without touching learned weights.

# Deep Active Inference for Autonomous Driving — Implementation Plan

## Overview

CARLA 시뮬레이터에서 Karl Friston의 능동추론(Active Inference) 이론에 기반한 자율주행 에이전트를 구현한다.
강화학습이 아닌, 생성 모델(Generative Model)을 학습하고 기대 자유 에너지(EFE)를 최소화하는 방향으로 행동을 추론한다.

## Confirmed Design Decisions

| 항목 | 결정 | 근거 |
|---|---|---|
| **Action Space** | 2D `[steer, accel]` | accel>0→throttle, accel<0→brake. CEM 탐색 40% 효율화 |
| **Latent Space** | Gaussian (64-dim) | Active Inference 이론 부합 (closed-form KL). logvar clamp [-20,2], min_std=0.1 |
| **KL Balancing** | Dual KL + free_nats=1.0 | dyn=1.0, rep=0.1, stop-gradient. DreamerV3 패턴 |
| **Epistemic Value** | Ensemble disagreement (primary) + decoder variance (ablation) | 5 MLP heads, `.mean()` 정규화 |
| **GMM Preference** | K=3, diagonal, gradient MLE | Oracle 권고: K=5는 샘플 부족. k-means++ 초기화, min_std=0.01 |
| **CEM Planner** | iCEM: 200 samples, 3 iters, 20 elites, H=12 | Colored noise β=1, warm-start. Batch-parallel rollout (~5-8ms) |
| **Decoder** | Mean-only output, MSE loss | DreamerV3 스타일. NLL 대신 per-dim MSE (÷C×H×W) |
| **CARLA** | 0.9.16, Town06 train, Town04/03 test | Town04/03은 failure mode analysis로 프레이밍 |
| **Web UI** | Streamlit | Training curves, t-SNE, imagination rollout |
| **Build** | Fresh in `active-inference-omo/`, uv | 기존 Antigravity 코드는 레퍼런스만 참조 |

## Architecture

```
State = { deter: h_t [B,256] (GRU), stoch: z_t [B,64] (Gaussian) }
feat  = concat(deter, stoch) = [B, 320]  ← 모든 Head의 입력

obs_step(prev_state, prev_action, embed):   # 학습 — 관측 있음
  prior = img_step(prev_state, prev_action)
  x = cat(prior.deter, embed)
  post_stats = MLP(x) → (mean, std)
  z = Normal(mean, std).rsample()
  return post, prior

img_step(prev_state, prev_action):           # CEM 상상 — 관측 없음
  x = cat(prev_state.stoch, prev_action)
  h = GRU(MLP(x), prev_state.deter)
  prior_stats = MLP(h) → (mean, std)
  z = Normal(mean, std).rsample()
  return {deter=h, stoch=z}

VFE = per_dim_MSE(recon) + 1.0*max(KL(sg(post)||prior), 1.0) + 0.1*max(KL(post||sg(prior)), 1.0)
EFE = β_i * MC_KL[q(s|a)||GMM] - β_e * ensemble_disagreement.mean(dim=-1)
```

## Module Dependency Graph

```
T1: Foundation ─────┬──── T2: RSSM ────┬── T3: VFE Loss
                    │                   ├── T4: Ensemble
                    │                   ├── T5: GMM Preference
                    │                   │       │
                    │                   │       ├── T6: EFE ── T7: CEM ──┐
                    │                   │       │                         │
                    ├── T8: Dataset ────┼───────┼─────────────── T9: Agent
                    │                   │       │                    │
                    └── T10: CARLA Env ─┼───────┼── T11: Collect ─── T12: Train
                                        │       │                    │
                                        │       └── T13: Evaluate ───┤
                                        │                            │
                                        │           T14: E2E ────────┘
                                        │                │
                                        └── T15: Dashboard  T16: Viz  T17: Live UI
```

---

## Wave-Based Execution Plan (17 Tasks, 9 Waves)

### Wave 1 — Foundation (No Dependencies)

#### T1: Project Skeleton + Dependencies + Config
- **Category**: `quick` | **Skills**: `[git-master]`
- **What**:
  - `pyproject.toml`: uv, torch (CUDA 12.6 index routing), carla==0.9.16, numpy, scipy, h5py, tensorboard, wandb, streamlit, omegaconf, tqdm, rich, einops, opencv-python, imageio
  - Dev: pytest, pytest-cov, ruff, mypy
  - `src/active_inference/` package: `models/`, `planning/`, `training/`, `evaluation/`, `data/`, `ui/`, `utils/`
  - `utils/transforms.py`: `symlog(x) = sign(x)*log(|x|+1)`, `symexp`, `normalize_image`, `denormalize_image`
  - `utils/seed.py`: global seed setting (torch, numpy, random)
  - `configs/default.yaml`: all hyperparameters
  - `configs/experiment/debug.yaml`: stoch=8, deter=32, 2 epochs, batch=4
  - `config.py`: `@dataclass Config` with `from_yaml()`, validation
  - `.gitignore`, `.python-version`
- **QA**: `uv sync && python -c "from active_inference.config import Config; c = Config.from_yaml('configs/default.yaml'); assert c.rssm.stoch_dim == 64"`

### Wave 2 — Core Models + Data (After T1, parallel)

#### T2: RSSM Model
- **Category**: `deep` | **Skills**: `[]`
- **What**:
  - `models/encoder.py`: ConvEncoder — 4-layer CNN (64×64→4096→256) + proprioceptive MLP (2→64→256) + fusion (512→256)
  - `models/decoder.py`: ObsDecoder — deconv (mean only), StateDecoder — MLP for speed/steer
  - `models/rssm.py`: RSSM class
    - `initial(batch_size)` → zero state
    - `obs_step(prev_state, prev_action, embed)` → post, prior
    - `img_step(prev_state, prev_action)` → prior
    - `imagine(initial_state, actions)` → trajectory (batch-parallel: accepts `[n_samples, H, action_dim]`)
    - `get_feat(state)` → `[B, 320]`
    - `get_dist(state)` → `Normal(mean, std)`
    - logvar clamp `[-20, 2]`, min_std=0.1
- **Tests** (8):
  - `test_rssm_hidden_state_changes`: h after sequence ≠ h_0
  - `test_obs_step_differs_from_img_step`: posterior ≠ prior
  - `test_imagine_trajectory_shapes`: `[H, B, 320]` 출력
  - `test_get_feat_shape`: `[B, 320]`
  - `test_rssm_gru_input_composition`: prev_action 변경 → GRU 출력 변경
  - `test_rssm_initial_state`: zeros 반환
  - `test_encoder_output_shape`: `[B, 256]`
  - `test_decoder_output_shape`: 이미지 `[B, 3, 64, 64]`, 상태 `[B, 2]`
- **QA**: `pytest tests/test_rssm.py tests/test_encoder.py tests/test_decoder.py -v`

#### T8: Synthetic Data + Dataset
- **Category**: `unspecified-low` | **Skills**: `[]`
- **What**:
  - `data/synthetic.py`: SyntheticDrivingData
    - `generate(n_episodes, episode_len)` → sinusoidal steer, constant speed + noise, Gaussian images
    - `to_hdf5(path)` / `from_hdf5(path)` — keys: images, states, actions, episode_ids
  - `data/dataset.py`: SequenceDataset(torch.utils.data.Dataset)
    - HDF5 로딩, 고정 길이 시퀀스 반환
    - episode_ids 기반 에피소드 경계 처리 (경계 넘는 시퀀스 제외)
    - `get_dataloader()` 팩토리 함수
- **Tests** (5):
  - `test_synthetic_generation`: 올바른 shape, NaN 없음
  - `test_hdf5_roundtrip`: 저장→로드 일치
  - `test_dataset_sequence_extraction`: seq_len 길이 시퀀스 반환
  - `test_dataset_episode_boundaries`: 에피소드 경계 넘는 시퀀스 제외됨
  - `test_dataloader_batching`: DataLoader에서 올바른 batch shape
- **QA**: `pytest tests/test_synthetic.py tests/test_dataset.py -v`

#### T10: CARLA Environment Wrapper
- **Category**: `unspecified-high` | **Skills**: `[]`
- **What**:
  - `data/carla_env.py`: CARLADrivingEnv
    - Sync mode: `fixed_delta=0.05` (20Hz)
    - Camera: 256×256 capture → queue → resize 64×64 → `/255 - 0.5` 정규화
    - Collision sensor + lane invasion sensor (callback → flag)
    - `step(action_2d)` → obs, info
    - `reset(spawn_point)` → obs
    - `close()` — actor cleanup, sync 해제
    - `action_to_carla([steer, accel])` → `(steer, throttle, brake)`
    - `carla_to_action(steer, throttle, brake)` → `[steer, throttle - brake]`
- **Tests** (4, unit only — CARLA 불필요):
  - `test_action_to_carla`: accel=0.5→(0.5, 0), accel=-0.3→(0, 0.3)
  - `test_carla_to_action`: (0.1, 0.5, 0.0)→[0.1, 0.5]
  - `test_action_roundtrip`: to_carla → from_carla 일치
  - `test_carla_to_action_simultaneous`: (0, 0.5, 0.3)→[0, 0.2]
- **QA**: `pytest tests/test_carla_env.py -v -k "not integration"`

### Wave 3 — Loss + Components (After T2, parallel)

#### T3: VFE Loss + Transforms
- **Category**: `ultrabrain` | **Skills**: `[]`
- **What**:
  - `training/losses.py`: `compute_vfe(post, prior, obs_img, obs_state, recon_img, recon_state)`
    - Dual KL: `dyn_loss = KL(sg(post)||prior)`, `rep_loss = KL(post||sg(prior))`
    - Free nats: `torch.clamp(kl, min=1.0)`
    - Image recon: per-dim MSE `/ (C*H*W)`
    - State recon: MSE on `symlog(target)` vs `symlog(pred)`
    - Total: `recon + 1.0*dyn + 0.1*rep`
    - Returns: `(total_loss, {"nll": ..., "kl_dyn": ..., "kl_rep": ..., "kl_per_dim": ...})`
- **Tests** (7):
  - `test_vfe_known_analytical`: 두 알려진 Gaussian, KL ±1e-4
  - `test_stop_gradient_dyn`: `torch.autograd.grad(dyn_loss, post_params)` = None
  - `test_stop_gradient_rep`: `torch.autograd.grad(rep_loss, prior_params)` = None
  - `test_free_nats_clipping`: KL < 1.0 → 1.0으로 clamp
  - `test_loss_scale_balance`: recon과 KL이 10× 이내
  - `test_vfe_no_nan_with_small_std`: min_std 경계에서 NaN 없음
  - `test_symlog_symexp_inverse`: `symexp(symlog(x)) ≈ x`
- **QA**: `pytest tests/test_losses.py tests/test_transforms.py -v`

#### T4: Ensemble Transition Heads
- **Category**: `quick` | **Skills**: `[]`
- **What**:
  - `models/ensemble.py`: EnsembleTransitionHeads(nn.Module)
    - 5 MLP heads: `Linear(320, 256)→ReLU→Linear(256, 64*2)` → split mean, logvar
    - `forward(feat)` → stacked_means `[K,B,64]`, stacked_stds `[K,B,64]`
    - `epistemic_uncertainty(feat)` → `means.var(dim=0).mean(dim=-1)` `[B]` (`.mean()` 정규화)
    - Training: 같은 데이터, 다른 random init, RSSM과 jointly
- **Tests** (3):
  - `test_ensemble_k_outputs`: K=5 서로 다른 예측
  - `test_epistemic_positive`: 랜덤 입력에서 uncertainty > 0
  - `test_ensemble_shape`: 올바른 차원
- **QA**: `pytest tests/test_ensemble.py -v`

#### T5: GMM Preference Model
- **Category**: `unspecified-high` | **Skills**: `[]`
- **What**:
  - `training/preference.py`: PreferenceModel
    - K=3 diagonal Gaussian, `MixtureSameFamily`
    - means `[K,64]`, log_stds `[K,64]`, logits `[K]` — all `nn.Parameter`
    - `distribution()` → GMM
    - `log_prob(z)` → `[B]`
    - `update_from_trajectories(encoder, rssm, expert_sequences, n_iters=200, lr=0.01)`:
      1. Expert 이미지 시퀀스를 encoder → RSSM `obs_step` (전체 시퀀스, temporal context 포함)
      2. posterior mean 수집 (모든 timestep)
      3. k-means++ 초기화
      4. Gradient MLE, min_std=0.01
      5. Early stopping (log_prob 변화 < 1e-4)
    - `save_state()` / `load_state()`
    - 단일 공유 GMM (Task A+B)
- **Tests** (5):
  - `test_gmm_log_prob_higher_at_means`: log_prob(means) > log_prob(random)
  - `test_gmm_fit_recovers_known`: 8차원 테스트 데이터, ±1.0
  - `test_gmm_save_load_roundtrip`: 저장→로드→log_prob 일치
  - `test_gmm_update_uses_rssm_sequence`: mock RSSM, obs_step 호출 확인
  - `test_gmm_no_component_collapse`: 피팅 후 K개 mean 간 pairwise distance > threshold
- **QA**: `pytest tests/test_preference.py -v`

### Wave 4 — EFE (After T4 + T5)

#### T6: EFE Computation
- **Category**: `ultrabrain` | **Skills**: `[]`
- **What**:
  - `planning/efe.py`: EFEScorer
    - `instrumental_value(q_dist, pref_model, n_samples=32)`:
      `z = q_dist.rsample((32,))`, `mc_kl = (q.log_prob(z).sum(-1) - pref.log_prob(z)).mean(0)`
    - `epistemic_value_ensemble(ensemble, feat)`:
      `preds.var(dim=0).mean(dim=-1)` (`.mean()` 정규화)
    - `epistemic_value_decoder(decoder, z_dist, n_samples=10)`:
      decode each sample, `means.var(dim=0).sum(-1)`
    - `score(traj_feats, q_dists, pref, ensemble, mode='ensemble')`:
      `sum_over_horizon(β_i * instrumental - β_e * epistemic)`
    - `β_i=1.0, β_e=0.1` (configurable)
- **Tests** (4):
  - `test_efe_lower_near_gmm`: GMM mode 근처 state → 낮은 EFE
  - `test_epistemic_higher_uncertain`: untrained ensemble → high epistemic
  - `test_efe_gradient_flows`: EFE가 action에 대해 미분 가능
  - `test_score_shape`: `[B]` 출력
- **QA**: `pytest tests/test_efe.py -v`

### Wave 5 — CEM Planner (After T6)

#### T7: iCEM Planner
- **Category**: `deep` | **Skills**: `[]`
- **What**:
  - `planning/cem_planner.py`: iCEMPlanner
    - 200 samples, 3 iters, 20 elites, horizon=12
    - `_colored_noise(shape, beta)`: FFT 기반 1/f^β noise (Timmer & König 1995)
    - Warm-start: 이전 solution shift left + append zero
    - `plan(initial_state, rssm, efe_scorer, pref_model)` → `[2]` action
      - **Batch-parallel**: 200 samples를 하나의 batch로 `rssm.imagine()` 호출
      - Score by negative EFE, select top-20
    - `action_to_carla(action_2d)` → `(steer, throttle, brake)` static method
- **Tests** (7):
  - `test_cem_converges_quadratic`: `||a - a*||² → a*` within 0.1
  - `test_cem_bounds_respected`: 모든 출력 `[-1, 1]`
  - `test_warm_start_improves`: 2번째 호출 시 first-iter score 향상
  - `test_colored_noise_correlated`: lag-1 autocorrelation > 0.3
  - `test_colored_noise_shape`: `[n_samples, H, action_dim]`
  - `test_action_to_carla`: accel=0.5→(0.5,0), accel=-0.3→(0,0.3)
  - `test_cem_latency`: `plan()` < 200ms on CPU (GPU에서 ~5-8ms 예상)
- **QA**: `pytest tests/test_cem.py -v`

### Wave 6 — Agent Integration (After T7 + T8)

#### T9: DeepAIFAgent
- **Category**: `deep` | **Skills**: `[git-master]`
- **What**:
  - `agent.py`: DeepAIFAgent
    - Owns: encoder, decoder, rssm, ensemble (as `WorldModel` wrapper), cem_planner, preference_model
    - `step(obs_img [3,64,64], obs_state [2])` → `action_2d [2]`
    - `update(batch)` → loss dict
      - BPTT: `bptt_window=50` (config), hidden state `.detach()` at boundaries
      - Gradient clip `max_norm=100`
      - Episode boundary → hidden state reset (done flag)
    - `reset()` → zero states
    - `save_checkpoint(path)` / `load_checkpoint(path)` — model + optimizer + GMM + config
- **Tests** (7):
  - `test_forward_pass`: step → `[2]` tensor
  - `test_action_range`: 출력 `[-1, 1]`
  - `test_checkpoint_roundtrip`: save→load→step 동일 출력 (seed 고정)
  - `test_update_reduces_loss`: synthetic data 5 step → loss 감소
  - `test_hidden_state_detach`: update 후 gradient graph 크기 bounded
  - `test_episode_boundary_resets_hidden`: done=True → next hidden = zeros
  - `test_stoch_is_used`: stoch masking → reconstruction 품질 저하
- **QA**: `pytest tests/test_agent.py -v`

### Wave 7 — CARLA Pipelines (After T9 + T10, parallel)

#### T11: Data Collection Script
- **Category**: `unspecified-low` | **Skills**: `[]`
- **What**:
  - `scripts/collect_data.py`:
    - BasicAgent autopilot on Town06
    - HDF5: images `[N,3,64,64]`, states `[N,2]`, actions `[N,2]`, episode_ids `[N]`, metadata
    - `carla_to_action()` 변환으로 2D action 저장
    - Success 태깅: Task A = no collision + mean lateral_dev < 0.5m
    - NPC traffic 10-20대 spawn
    - CLI: `--town, --num_samples, --output`
- **QA**: `python scripts/collect_data.py --num_samples 100 --output data/test.h5` → valid HDF5 (CARLA 필요)

#### T12: Training Pipeline
- **Category**: `deep` | **Skills**: `[]`
- **What**:
  - `scripts/train.py`:
    - Config-driven (YAML → Config dataclass)
    - SequenceDataset + DataLoader
    - BPTT: window=50, hidden detach, episode boundary reset
    - GMM update: every epoch after epoch 10 (on success-tagged data)
    - Logging: TensorBoard + wandb — VFE, KL_dyn, KL_rep, NLL, `posterior_entropy`, `prior_entropy`, `kl_per_dim`
    - Collapse alert: `posterior_entropy < 0.1` → warning log
    - NaN detection: `torch.isnan(loss)` → error + checkpoint rollback
    - Checkpoint: model + optimizer + GMM + epoch + config
    - LR scheduler: warmup 1000 steps → constant
    - CLI: `--config, --data, --epochs, --device`
- **QA**: `python scripts/train.py --config debug.yaml --data synthetic.h5 --epochs 2` → loss(epoch2) < loss(epoch0), checkpoint 저장, NaN 없음

#### T13: Evaluation Pipeline
- **Category**: `unspecified-high` | **Skills**: `[]`
- **What**:
  - `scripts/evaluate.py`:
    - Task A: Town04 (curves), 5 episodes × 1000 frames
    - Task B: Town03 (obstacles), 5 episodes × 1000 frames + static obstacle spawn
    - 또한 Town06 (same-domain baseline) 추가 — Oracle 권고
    - Metrics: Success Rate, Mean Lateral Deviation, Off-Road Events
    - Video: imageio mp4 + telemetry overlay (speed, steer, EFE, epistemic score)
    - CSV: `eval_results.csv` — columns: `[task, town, episode, success, mean_lateral_dev, offroad_events, mean_epistemic]`
    - Epistemic annealing: `β_e = 0.1 × β_i` during eval
    - Safety fallback: epistemic > threshold → log warning (calibrate on Town06)
    - CLI: `--task, --checkpoint, --episodes, --save_video`
- **QA**: CSV 생성 + 올바른 컬럼, video file (if `--save_video`)

### Wave 8 — E2E Pipeline (After Wave 7)

#### T14: End-to-End Pipeline
- **Category**: `quick` | **Skills**: `[]`
- **What**: `scripts/run_pipeline.py` — collect(100) → train(2 epochs) → evaluate(1 episode)
- **QA**: 전체 파이프라인 에러 없이 완료, 모든 artifact 생성 (CARLA 필요)

### Wave 9 — Web UI (After Wave 7-8, parallel)

#### T15: Training Dashboard
- **Category**: `visual-engineering` | **Skills**: `[frontend-ui-ux]`
- **What**:
  - `ui/app.py`: Streamlit multi-page app
  - Page 1: Training curves (VFE, KL, NLL) — TensorBoard event 파싱 또는 CSV fallback
  - Page 2: Evaluation metrics table + bar chart (from `eval_results.csv`)
  - Auto-refresh
- **QA**: `timeout 10 streamlit run src/active_inference/ui/app.py --server.headless true` → 에러 없이 시작

#### T16: Visualization Pages
- **Category**: `visual-engineering` | **Skills**: `[frontend-ui-ux]`
- **What**:
  - Imagination Rollout: checkpoint 로드 → CEM planning states decode → 이미지 grid (H×iters)
  - Preference t-SNE: expert latents → t-SNE scatter + GMM contour overlay, Task A(blue)/B(red)
- **QA**: Python error 없이 페이지 렌더링

#### T17: Live Evaluation UI
- **Category**: `visual-engineering` | **Skills**: `[frontend-ui-ux, dev-browser]`
- **What**: 실행 중인 evaluation의 CARLA camera feed + real-time metrics + start/stop 컨트롤
- **QA**: UI 시작, mock 데이터 소스 연결 에러 없음

---

## Critical Path

```
T1 → T2 → T4/T5 → T6 → T7 → T9 → T12 → T14
```

Sequential 8 steps. Parallel waves로 전체 17 tasks 실행 시 ~45% 시간 단축.

## Risk Mitigation Summary

| Risk | Severity | Mitigation |
|---|---|---|
| Posterior collapse | High | Dual KL + free_nats + KL_per_dim 모니터링 + stoch-is-used 테스트 |
| CEM latency | High | Batch-parallel rollout (200→1 call) + latency 테스트 |
| GMM collapse | Medium | K=3, k-means++ init, min_std=0.01, component collapse 테스트 |
| Domain gap (Town04/03) | High | Failure analysis 프레이밍 + epistemic score 로깅 + Town06 baseline |
| Training data (72K) | Medium | Open-loop prediction 품질 모니터링 + holdout 2 episodes |
| BPTT OOM | Medium | bptt_window=50 + hidden detach |
| Action space mismatch | Low | `carla_to_action()` inverse + roundtrip 테스트 |

## Test Coverage vs Prior Bugs (17/17)

| Prior Bug | Test Coverage |
|---|---|
| BUG-1: Broken RSSM | `test_rssm_gru_input_composition`, `test_rssm_hidden_state_changes` |
| BUG-2: CEM no imagination | `test_imagine_trajectory_shapes`, batch-parallel rollout |
| BUG-3: GMM dummy zeros | `test_gmm_update_uses_rssm_sequence` |
| BUG-4: Invalid EFE approx | MC sampling (32 samples), `test_efe_lower_near_gmm` |
| BUG-5: No KL balancing | `test_stop_gradient_dyn`, `test_stop_gradient_rep`, `test_free_nats_clipping` |
| BUG-6: BPTT no detach | `test_hidden_state_detach` |
| BUG-7: logvar unbounded | logvar clamp [-20,2], `test_vfe_no_nan_with_small_std` |
| BUG-8: Loss scale imbalance | `test_loss_scale_balance` (recon÷C×H×W) |
| BUG-9: Wrong test towns | Town04/03 명시, Town06 baseline 추가 |
| BUG-10: Resolution mismatch | 64×64 통일, `test_encoder_output_shape` |
| BUG-11: No decoder variance | Ensemble epistemic (primary) + decoder variance (ablation) |
| BUG-12: Throttle+brake conflict | 2D action space, `test_action_to_carla` |
| BUG-13: GMM not saved | `test_checkpoint_roundtrip` (GMM 포함) |
| BUG-14: sys.path hacks | uv + installable package |
| BUG-15: No normalization | `normalize_image`, symlog in transforms.py |
| BUG-16: Stub update() | Full BPTT implementation in agent.update() |
| BUG-17: No collection sensors | Collision + lane invasion sensors in CARLA env |

## Config Schema (default.yaml)

```yaml
seed: 42
device: cuda

rssm:
  deter_dim: 256
  stoch_dim: 64
  embed_dim: 256
  min_std: 0.1
  logvar_clip: [-20, 2]

encoder:
  image_channels: 3
  image_size: 64
  state_dim: 2

training:
  lr: 1e-4
  epochs: 50
  batch_size: 32
  seq_len: 50
  bptt_window: 50
  grad_clip: 100.0
  free_nats: 1.0
  kl_dyn_scale: 1.0
  kl_rep_scale: 0.1
  lr_warmup_steps: 1000

ensemble:
  num_heads: 5
  hidden_dim: 256

cem:
  horizon: 12
  n_samples: 200
  n_elites: 20
  n_iters: 3
  colored_noise_beta: 1.0
  warm_start: true

preference:
  K: 3
  min_std: 0.01
  update_interval: 1  # every epoch after warmup
  warmup_epoch: 10
  fit_iters: 200
  fit_lr: 0.01

efe:
  beta_instrumental: 1.0
  beta_epistemic: 0.1
  mc_samples: 32

evaluation:
  episodes: 5
  max_frames: 1000
  epistemic_anneal: 0.1
  towns:
    task_a: Town04
    task_b: Town03
    baseline: Town06

data:
  carla_version: "0.9.16"
  collection_town: Town06
  num_samples: 72000
  fps: 20
  image_capture_size: 256
  image_model_size: 64
```

## Reference Implementations

| Component | Reference | URL |
|---|---|---|
| RSSM | DreamerV3-Torch | `NM512/dreamerv3-torch` networks.py |
| CEM Planner | PlaNet (Kaixhin) | `Kaixhin/PlaNet` planner.py |
| EFE | SR-AIF | `NACLab/self-revising-active-inference` agent.py |
| Active Inference | Fountas Deep-AIF-MC | `zfountas/deep-active-inference-mc` torchmodel.py |
| CARLA Env | CarDreamer | `ucd-dare/CarDreamer` carla_base_env.py |
| Evaluation | Bench2Drive/LEAD | `autonomousvision/lead` atomic_criteria.py |

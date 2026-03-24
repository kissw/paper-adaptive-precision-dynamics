# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Deep Active Inference agent for autonomous driving in CARLA 0.9.16. Uses Expected Free Energy (EFE) minimization with iCEM planning — **not** reinforcement learning. ALL control must come from the Active Inference framework. No classical controllers (Stanley, PID).

## Commands

```bash
# CARLA location 
/data/jaerock/carla-0.9.16 or ~/carla-0.9.16

# Install dependencies
uv sync

# Run all tests (72 tests, ~40s)
uv run pytest tests/ -v --tb=short

# Run a single test file or test
uv run pytest tests/test_efe.py -v
uv run pytest tests/test_agent.py::test_forward_pass -v

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Type check
uv run mypy src/

# Training
uv run python scripts/train.py --data data/expert_data_mixed.h5 --output outputs/train_v5

# Evaluation (requires running CARLA server on port 2000)
uv run python scripts/evaluate.py --task A --checkpoint outputs/train_v5_finetune/checkpoints/best.pt

# Data collection (requires CARLA)
uv run python scripts/collect_data.py --town Town06 --num_samples 72000 --output data/expert_data.h5

# Merge datasets
uv run python scripts/merge_data.py data/file1.h5 data/file2.h5 --output data/merged.h5

# Refit preference model (GMM only, world model frozen)
uv run python scripts/refit_preference.py --checkpoint path/to/best.pt --data data/expert_data.h5 --output path/to/refit.pt
```

## Architecture

The agent loop: **Observe → Encode → RSSM posterior → iCEM plan (EFE scoring) → Act**

```
DeepAIFAgent (agent.py)
├── WorldModel
│   ├── ConvEncoder: 64×64 RGB + 4D proprioceptive state → 256D embedding
│   ├── RSSM: GRU(deter=256) + Gaussian(stoch=64), obs_step/img_step/imagine
│   ├── ObsDecoder: 320D features → 64×64 image reconstruction
│   ├── StateDecoder: 320D → [speed, steer, heading_error, crosstrack_error]
│   └── EnsembleTransitionHeads: 5 MLP heads for epistemic uncertainty
├── PreferenceModel: GMM (K=5, 64D) fitted on expert latents via MLE
├── EFEScorer: instrumental (GMM cross-entropy) + epistemic (ensemble) + state-space penalty
└── iCEMPlanner: 500 samples, 50 elites, 5 iters, horizon=12, colored noise
```

**Training** minimizes Variational Free Energy (VFE): image MSE + state MSE (symlog) + dual KL (DreamerV3 style). Per-timestep backward with AMP, NaN guard, CUDA recovery.

**Planning** uses iCEM with warm-start, cold-start recovery (extra iters + doubled samples), and adaptive precision-weighting (state error modulates exploration width and EFE channel weights).

## Key Design Decisions

- **State decoder predictions during imagination are noisy** — state penalties use raw values with `clamp(max=4.0)`, never z-scored. `beta_state=0.3` works; `beta_state=1.0` with z-scoring crashes.
- **Action space**: `[steer, accel]` ∈ [-1, 1]. Accel maps linearly to throttle ∈ [0.35, 0.55]. No braking.
- **Accel prior**: CEM initializes accel channel to 0.3 to break cold-start symmetry.
- **4D state**: `[speed_mps, steer, heading_error, crosstrack_error]`. Encoder uses all 4; state decoder reconstructs all 4.
- **Two tasks**: Task A = lane keeping (Town04, crosstrack≈0). Task B = obstacle avoidance via lane change (Town06_Opt, crosstrack shifts during maneuver). Separate preference models, shared world model.

## Config System

Dataclass-based (`src/active_inference/config.py`) with YAML overlay via OmegaConf. `Config.from_yaml()` merges schema defaults with YAML overrides.

- `configs/default.yaml` — production config
- `configs/experiment/debug.yaml` — tiny model for fast tests
- `configs/experiment/task_b.yaml` — Task B overrides (lower beta_state, longer horizon, more GMM components)

## Data Format

HDF5 files with datasets: `images` [N,3,64,64], `states` [N,4], `actions` [N,2], `episode_ids`, `success_flags`, `task_labels` (0=lane-keep, 1=lane-change), `lane_ids`, `lateral_devs`, `noise_sigmas`.

## Testing

Tests use synthetic data (`SyntheticDrivingData`) and the debug config — no CARLA needed. Integration tests marked with `@pytest.mark.integration`. Test files mirror source structure: `test_efe.py` tests `planning/efe.py`, etc.

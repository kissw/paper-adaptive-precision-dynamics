# Active Inference for Autonomous Driving

Deep Active Inference agent for autonomous driving in CARLA 0.9.16. Uses Expected Free Energy (EFE) minimization with iCEM planning to learn steering control and obstacle avoidance. All control emerges from the Active Inference framework — no classical controllers (Stanley, PID). The agent learns to drive by minimizing the divergence between predicted and preferred future states, analogous to how humans acquire motor skills through motor babbling and proprioceptive feedback.

## Key Results

| Task | Metric | Result |
|------|--------|--------|
| **A: Lane Keeping** | Completion (moderate curves, 0°→85°) | **81.2%** |
| | Heading-steer correlation | r = 0.41 (learned reactive steering) |
| | Mean lateral deviation | 0.624m |
| **B: Obstacle Avoidance** | Success rate (8 episodes, 2 routes) | **100%** |
| | Obstacles avoided | **19/19 (100%)** |
| | Mean lateral deviation | 0.66m |

## Architecture

```
Observe → Encode → RSSM Posterior → iCEM Plan (EFE Scoring) → Act

DeepAIFAgent
├── WorldModel
│   ├── ConvEncoder: 64×64 RGB + 4D state → 256D embedding
│   ├── RSSM: GRU(deter=256) + Gaussian(stoch=64)
│   ├── ObsDecoder: 320D features → 64×64 image
│   ├── StateDecoder: 320D → [speed, steer, heading_error, crosstrack_error]
│   └── EnsembleTransitionHeads: 5 MLP heads (epistemic uncertainty)
├── PreferenceModel: GMM (K=5/7, 64D latent) fitted on expert data
├── EFEScorer: instrumental (GMM) + epistemic (ensemble) + state penalty
└── iCEMPlanner: 500 samples, 50 elites, 5 iters, colored noise
```

## Quick Start

### Prerequisites

- Python 3.10+
- CARLA 0.9.16 (at `/data/jaerock/carla-0.9.16` or `~/carla-0.9.16`)
- CUDA-capable GPU

### Install

```bash
uv sync
```

### Train

```bash
uv run python scripts/train.py --data data/expert_data_mixed.h5 --output outputs/train_v5
```

### Evaluate

```bash
# Start CARLA server first (port 2000)

# Task A: Lane keeping (Town04, moderate curves)
uv run python scripts/evaluate.py --task A --checkpoint outputs/train_v5_finetune/checkpoints/best.pt

# Task B: Obstacle avoidance (Town06_Opt)
uv run python scripts/evaluate.py --task B --checkpoint outputs/train_v5_finetune/checkpoints/best.pt \
    --config configs/experiment/task_b_v5.yaml
```

### Test

```bash
uv run pytest tests/ -v --tb=short    # 72 tests, ~40s, no CARLA needed
uv run ruff check src/ tests/         # Lint
uv run mypy src/                      # Type check
```

## Project Structure

```
src/active_inference/
├── agent.py                    # DeepAIFAgent, WorldModel
├── config.py                   # Dataclass config + YAML overlay (OmegaConf)
├── models/
│   ├── rssm.py                 # Recurrent State-Space Model
│   ├── encoder.py              # ConvEncoder (image + state fusion)
│   ├── decoder.py              # ObsDecoder (image), StateDecoder (4D state)
│   └── ensemble.py             # EnsembleTransitionHeads (epistemic uncertainty)
├── planning/
│   ├── cem_planner.py          # iCEM planner (colored noise, warm/cold start)
│   └── efe.py                  # EFE scorer (instrumental + epistemic + state)
├── training/
│   ├── losses.py               # VFE loss (image + state + dual KL)
│   └── preference.py           # GMM preference model
├── evaluation/
│   ├── routes.py               # CARLA evaluation routes
│   └── obstacles.py            # Obstacle spawning for Task B
├── data/
│   ├── dataset.py              # HDF5 dataset loader
│   ├── carla_env.py            # CARLA driving environment
│   └── synthetic.py            # Synthetic data for unit tests
└── utils/
    ├── transforms.py           # Image transforms
    └── seed.py                 # Reproducibility

configs/
├── default.yaml                # Production config (Task A)
└── experiment/
    ├── debug.yaml              # Tiny model for tests
    ├── task_b.yaml             # Task B (deprecated)
    └── task_b_v5.yaml          # Task B v5 obstacle avoidance (active)

scripts/
├── train.py                    # Training script
├── evaluate.py                 # Evaluation with obstacle detection
├── collect_data.py             # Expert data collection from CARLA
├── merge_data.py               # HDF5 dataset merging
└── refit_preference.py         # Task-specific GMM preference fitting
```

## Documentation

- **[System Design](docs/system-design.md)** — architecture, EFE formulation, training and planning pipelines
- **[Experiment Results](docs/experiment-results.md)** — full results for Task A and Task B with version progression

# Adaptive Precision Dynamics in Deep Active Inference: Emerging Driving Behaviors from a Single World Model

**Jaerock Kwon¹, Elahe Delavari¹, Donghyun Kim², Haewoon Nam²**

¹ University of Michigan–Dearborn · ² Hanyang University

[![Paper](https://img.shields.io/badge/Paper-EIML%40ICML2026-blue)](paper/eiml2026/main.pdf)
[![CARLA](https://img.shields.io/badge/Simulator-CARLA%200.9.16-green)](https://carla.org/)

## Abstract

We present a deep active inference agent where adaptive precision-weighting enables lane keeping, obstacle avoidance, and post-evasion centering to emerge from a single Expected Free Energy (EFE) objective and shared RSSM world model. By dynamically modulating EFE component weights based on sensory prediction errors, the agent achieves 81.2% route completion on curves and 100% obstacle avoidance (28/28) in CARLA — without separate reward functions or control modules. We further investigate whether obstacle detection can be learned from the latent space alone, discovering a **posterior-prior epistemic gap**: an auxiliary prediction head achieves perfect obstacle encoding in the posterior (276 nats separation), yet avoidance remains at 0% because the RSSM transition model cannot propagate obstacle information through action-conditioned imagination.

## Key Results

| Task | Metric | Result |
|------|--------|--------|
| **Task A: Lane Keeping** | Route completion (moderate curves) | 81.2% |
| | Mean lateral deviation | 0.63m |
| **Task B: Obstacle Avoidance** | Success rate | 100% |
| | Obstacles avoided | 28/28 |
| **Post-Evasion Centering** | Active lane changes per episode | 6 |

## Architecture

The agent operates in a closed loop: **Observe → Encode → RSSM Posterior → iCEM Plan (EFE Scoring) → Act**

```
DeepAIFAgent
├── WorldModel
│   ├── ConvEncoder: 64×64 RGB + 4D state → 256D embedding
│   ├── RSSM: GRU(deter=256) + Gaussian(stoch=64)
│   ├── ObsDecoder: 320D features → 64×64 image
│   ├── StateDecoder: 320D → [speed, steer, heading_error, crosstrack_error]
│   └── EnsembleTransitionHeads: 5 MLP heads for epistemic uncertainty
├── PreferenceModel: GMM (K=5, 64D) fitted on expert latents
├── EFEScorer: β_i·instrumental + β_e·epistemic + β_s·state + β_o·obstacle
└── iCEMPlanner: 500 samples, 50 elites, 5 iters, horizon 12-15
```

## Adaptive Precision Dynamics

The core contribution: **same EFE objective, different precision weights → different behaviors**.

| Parameter | Task A (Lane Keeping) | Task B (Obstacle Avoidance) |
|-----------|----------------------|---------------------------|
| β_instrumental | 1.0 | 1.0 |
| β_epistemic | 0.1 | 0.1 |
| β_state | 0.5 | 0.0 |
| β_obstacle | 0.0 | 40.0 |
| Horizon | 12 | 15 |

Runtime modulation:
- **State error** > 0.3 → increase β_s (trust state decoder), decrease β_i (distrust GMM)
- **Obstacle proximity** > 0.1 → suppress β_i (let obstacle penalty dominate)
- **Post-evasion** → temporary β_s = 0.5 (pull back to lane center)

## Posterior-Prior Epistemic Gap

Our investigation into pure vision-based obstacle avoidance reveals:

| Method | Posterior Gap | Avoidance |
|--------|-------------|-----------|
| Single GMM (64×64) | −24.6 nats | 0% |
| Single GMM (128×128) | −2.3 nats | 0% |
| Contrastive preference | +37.0 nats | 0% |
| **Auxiliary head** | **+276 nats** | **0%** |

The RSSM posterior encodes obstacles perfectly, but the transition model cannot propagate this through imagination — a fundamental limitation of current RSSM-based planning.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests (72 tests)
uv run pytest tests/ -v --tb=short

# Training
uv run python scripts/train.py --data data/expert_data_mixed.h5 --output outputs/train

# Evaluation (requires CARLA 0.9.16 on port 2000)
uv run python scripts/evaluate.py --task A --checkpoint outputs/train/checkpoints/best.pt
uv run python scripts/evaluate.py --task B --checkpoint outputs/train/checkpoints/best.pt \
    --config configs/experiment/task_b_v5.yaml --route_index 1 --episodes 3
```

## Citation

```bibtex
@inproceedings{kwon2026adaptive,
  title={Adaptive Precision Dynamics in Deep Active Inference: Emerging Driving Behaviors from a Single World Model},
  author={Kwon, Jaerock and Delavari, Elahe and Kim, Donghyun and Nam, Haewoon},
  booktitle={EIML Workshop at ICML},
  year={2026}
}
```

## License

MIT

## Contact

- Jaerock Kwon — jrkwon@umich.edu
- Bio-Inspired Machine Intelligence (BIMI) Lab, University of Michigan–Dearborn

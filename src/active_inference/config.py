from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class RSSMConfig:
    deter_dim: int = 256
    stoch_dim: int = 64
    embed_dim: int = 256
    min_std: float = 0.1
    logvar_clip_low: float = -20.0
    logvar_clip_high: float = 2.0


@dataclass
class EncoderConfig:
    image_channels: int = 3
    image_size: int = 64
    state_dim: int = 2
    crop_road: bool = False


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    seq_len: int = 50
    bptt_window: int = 50
    grad_clip: float = 100.0
    free_nats: float = 1.0
    kl_dyn_scale: float = 1.0
    kl_rep_scale: float = 0.1
    lr_warmup_steps: int = 1000
    beta_obstacle_aux: float = 0.0  # auxiliary obstacle prediction loss weight


@dataclass
class EnsembleConfig:
    num_heads: int = 5
    hidden_dim: int = 256


@dataclass
class CEMConfig:
    horizon: int = 12
    n_samples: int = 200
    n_elites: int = 20
    n_iters: int = 3
    colored_noise_beta: float = 1.0
    warm_start: bool = True
    action_dim: int = 2
    noise_scale: list[float] | None = None
    accel_prior: float = 0.3
    cold_start_extra_iters: int = 5
    min_std: float = 0.1
    keep_fraction: float = 0.1
    warm_start_reset_threshold: float = 1.0


@dataclass
class PreferenceConfig:
    K: int = 3
    min_std: float = 0.01
    update_interval: int = 1
    warmup_epoch: int = 10
    fit_iters: int = 200
    fit_lr: float = 0.01
    data_path: str | None = None
    task_b_data: str | None = None
    max_samples: int = 3000
    balance_ratio: float = 0.5


@dataclass
class EFEConfig:
    beta_instrumental: float = 1.0
    beta_epistemic: float = 0.1
    beta_state: float = 1.0
    beta_obstacle: float = 0.0
    mc_samples: int = 32
    temporal_discount: float = 0.95
    heading_only_state: bool = False


@dataclass
class EvalTowns:
    task_a: str = "Town04"
    task_b: str = "Town06_Opt"
    baseline: str = "Town06"


@dataclass
class EvaluationConfig:
    episodes: int = 5
    max_frames: int = 1000
    epistemic_anneal: float = 0.1
    towns: EvalTowns = field(default_factory=EvalTowns)


@dataclass
class DataConfig:
    carla_version: str = "0.9.16"
    collection_town: str = "Town06"
    num_samples: int = 72000
    fps: int = 20
    image_capture_size: int = 256
    image_model_size: int = 64


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    cem: CEMConfig = field(default_factory=CEMConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    efe: EFEConfig = field(default_factory=EFEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        schema = OmegaConf.structured(cls)
        raw = OmegaConf.load(path)
        merged = OmegaConf.merge(schema, raw)
        return OmegaConf.to_object(merged)

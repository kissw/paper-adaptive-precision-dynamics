import torch
from torch import Tensor
from torch.distributions import Normal

from active_inference.models.ensemble import EnsembleTransitionHeads
from active_inference.training.preference import PreferenceModel


class EFEScorer:
    def __init__(
        self,
        beta_instrumental: float = 1.0,
        beta_epistemic: float = 0.1,
        mc_samples: int = 32,
        temporal_discount: float = 0.95,
    ):
        self._beta_i = beta_instrumental
        self._beta_e = beta_epistemic
        self._mc_samples = mc_samples
        self._gamma = temporal_discount

    def instrumental_value(
        self, q_mean: Tensor, q_std: Tensor, pref_model: PreferenceModel
    ) -> Tensor:
        # Cross-entropy E_q[-log p_pref(z)].
        # Using cross-entropy instead of full KL because the entropy term
        # H(q) is approximately constant w.r.t. action selection.
        # Raw cross-entropy preserves the full gradient signal for CEM.
        # q_mean, q_std: [B, D]
        q = Normal(q_mean, q_std)
        z = q.rsample((self._mc_samples,))  # [S, B, D]
        log_p = pref_model.log_prob(z)  # [S, B]
        return -log_p.mean(0)  # [B]

    def epistemic_value_ensemble(self, ensemble: EnsembleTransitionHeads, feat: Tensor) -> Tensor:
        return ensemble.epistemic_uncertainty(feat)  # [B]

    def epistemic_value_decoder(
        self, decoder, z_mean: Tensor, z_std: Tensor, n_samples: int = 10
    ) -> Tensor:
        # Decoder variance via MC sampling
        q = Normal(z_mean, z_std)
        samples = q.rsample((n_samples,))  # [S, B, D]
        decoded = torch.stack([decoder(s) for s in samples])  # [S, B, C, H, W]
        return decoded.var(dim=0).sum(dim=(1, 2, 3))  # [B]

    def score(
        self,
        trajectory_feats: list[Tensor],
        trajectory_means: list[Tensor],
        trajectory_stds: list[Tensor],
        pref_model: PreferenceModel,
        ensemble: EnsembleTransitionHeads,
    ) -> Tensor:
        # trajectory_feats: list of [B, feat_dim], len=horizon
        # trajectory_means: list of [B, stoch_dim]
        # trajectory_stds: list of [B, stoch_dim]
        # Returns: [B] total EFE (lower = better)
        B = trajectory_feats[0].shape[0]
        device = trajectory_feats[0].device
        total = torch.zeros(B, device=device)
        for t, (feat, mean, std) in enumerate(
            zip(trajectory_feats, trajectory_means, trajectory_stds)
        ):
            instr = self.instrumental_value(mean, std, pref_model)
            epist = self.epistemic_value_ensemble(ensemble, feat)
            # Per-timestep normalization: z-score across batch so relative
            # differences matter regardless of absolute magnitude.
            # Only normalize with sufficient samples (CEM uses B=500).
            if B > 4:
                instr = (instr - instr.mean().detach()) / (instr.std().detach() + 1e-8)
                epist = (epist - epist.mean().detach()) / (epist.std().detach() + 1e-8)
            discount = self._gamma ** t
            total = total + discount * (self._beta_i * instr - self._beta_e * epist)
        return total

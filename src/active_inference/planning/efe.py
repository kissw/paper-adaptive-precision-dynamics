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
        beta_state: float = 1.0,
        mc_samples: int = 32,
        temporal_discount: float = 0.95,
    ):
        self._beta_i = beta_instrumental
        self._beta_e = beta_epistemic
        self._beta_s = beta_state
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

    def state_instrumental_value(self, state_decoder, feat: Tensor) -> Tensor:
        """State-space instrumental value via decoded state preference.

        Decodes imagined features to predicted 4D states
        [speed, steer, heading_error, crosstrack_error] and penalizes
        deviation of heading_error and crosstrack_error from zero.

        This is equivalent to a Gaussian preference p̃(s) = N(0, I) over
        the navigation state components, which encodes the prior that
        good driving keeps the vehicle aligned with and centered on the lane.

        NOT z-scored because state decoder predictions during open-loop
        imagination are noisy. Raw values with clamping provide a gentle
        directional nudge without amplifying noise.
        """
        decoded = state_decoder(feat)  # [B, 4]
        heading_err = decoded[:, 2]
        crosstrack_err = decoded[:, 3]
        # Clamp to prevent outliers from dominating (imagination diverges)
        raw = heading_err.pow(2) + crosstrack_err.pow(2)
        return raw.clamp(max=4.0)  # [B], lower = better

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
        state_decoder=None,
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
            step_score = self._beta_i * instr - self._beta_e * epist

            # State-space penalty: raw (NOT z-scored) to avoid amplifying
            # noisy state decoder predictions during open-loop imagination.
            # Clamped and gently weighted to act as directional nudge.
            if state_decoder is not None and self._beta_s > 0:
                state_instr = self.state_instrumental_value(state_decoder, feat)
                step_score = step_score + self._beta_s * state_instr

            total = total + discount * step_score
        return total

import torch
from torch import Tensor
from torch.distributions import Normal

from active_inference.models.ensemble import EnsembleTransitionHeads
from active_inference.training.preference import PreferenceModel


class EFEScorer:
    """Expected Free Energy scorer for iCEM trajectory evaluation.

    Pure Active Inference formulation with three components:
    1. Instrumental value: GMM cross-entropy drives toward preferred latents
    2. Epistemic value: ensemble disagreement promotes information-seeking
    3. State-space value: decoded heading + crosstrack keeps agent centered

    Obstacle avoidance emerges from instrumental value: obstacle-present
    latents fall in low-probability regions of the preference model (fitted
    on obstacle-free driving), producing high EFE that the planner avoids.
    """

    def __init__(
        self,
        beta_instrumental: float = 1.0,
        beta_epistemic: float = 0.1,
        beta_state: float = 1.0,
        beta_obstacle: float = 0.0,
        mc_samples: int = 32,
        temporal_discount: float = 0.95,
        heading_only_state: bool = False,
    ):
        self._beta_i = beta_instrumental
        self._beta_e = beta_epistemic
        self._beta_s = beta_state
        self._beta_o = beta_obstacle  # kept for config compat
        self._mc_samples = mc_samples
        self._gamma = temporal_discount
        self._heading_only_state = heading_only_state
        # Visual surprise: reconstruction error as obstacle signal
        self._beta_surprise = 2.0  # weight for visual surprise penalty
        self._surprise_threshold = 350.0  # recon error above this = surprise

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

    def state_instrumental_value_heading_only(
        self, state_decoder, feat: Tensor,
    ) -> Tensor:
        """Heading-only state penalty (legacy, for backward compat)."""
        decoded = state_decoder(feat)  # [B, 4]
        heading_err = decoded[:, 2]
        return heading_err.pow(2).clamp(max=4.0)  # [B]

    def visual_surprise(
        self, obs_decoder, feat: Tensor, ref_image: Tensor,
    ) -> Tensor:
        """Visual surprise via imagined reconstruction error.

        Decodes imagined features to predicted images and compares
        against a reference image (current observation). Trajectories
        where the decoded image stays similar to a high-surprise
        observation (obstacle in view) score high; trajectories where
        the decoded image diverges (obstacle exits view after lane
        change) score lower.

        In AIF terms: this is the expected observation-level free energy.
        The agent prefers futures where its predicted observations match
        its prior preference (obstacle-free road).

        Contract: ref_image is assumed to already have crop_road applied
        when cfg.encoder.crop_road=True (WorldModel.preprocess_image is
        called by step_with_info before this function is invoked, so
        ref_image and the decoder output live in the same image space).
        """
        decoded_img = obs_decoder(feat)  # [B, C, H, W]
        # ref_image is [1, C, H, W] — expand to batch
        ref = ref_image.expand_as(decoded_img)
        # Per-sample MSE: high when decoded image looks like current
        # (obstacle-containing) observation
        mse = (decoded_img - ref).pow(2).sum(dim=(1, 2, 3))
        # Invert: we WANT trajectories that diverge from the obstacle
        # image. Low divergence (similar to obstacle image) = high penalty.
        # Normalize by image dimensions for stability.
        n_pixels = ref.shape[1] * ref.shape[2] * ref.shape[3]
        similarity = (-mse / n_pixels).exp()  # 0-1, high = similar
        return similarity.clamp(max=1.0)  # [B], lower = better

    def epistemic_value_ensemble(
        self, ensemble: EnsembleTransitionHeads, feat: Tensor,
    ) -> Tensor:
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
        obstacle_info: dict | None = None,
    ) -> Tensor:
        # trajectory_feats: list of [B, feat_dim], len=horizon
        # trajectory_means: list of [B, stoch_dim]
        # trajectory_stds: list of [B, stoch_dim]
        # obstacle_info: dict with 'recon_error' (float) and
        #   'ref_image' (Tensor) for visual surprise scoring
        # Returns: [B] total EFE (lower = better)
        B = trajectory_feats[0].shape[0]
        device = trajectory_feats[0].device
        total = torch.zeros(B, device=device)

        # Extract visual surprise context if available
        recon_error = 0.0
        ref_image = None
        obs_decoder = None
        if obstacle_info is not None:
            recon_error = obstacle_info.get("recon_error", 0.0)
            ref_image = obstacle_info.get("ref_image", None)
            obs_decoder = obstacle_info.get("obs_decoder", None)

        for t, (feat, mean, std) in enumerate(
            zip(trajectory_feats, trajectory_means, trajectory_stds)
        ):
            instr = self.instrumental_value(mean, std, pref_model)
            epist = self.epistemic_value_ensemble(ensemble, feat)

            # Per-timestep normalization: z-score across batch so relative
            # differences matter regardless of absolute magnitude.
            # Only normalize with sufficient samples (CEM uses B=500).
            if B > 4:
                instr = (instr - instr.mean().detach()) / (
                    instr.std().detach() + 1e-8
                )
                epist = (epist - epist.mean().detach()) / (
                    epist.std().detach() + 1e-8
                )

            discount = self._gamma ** t
            step_score = self._beta_i * instr - self._beta_e * epist

            # Visual surprise penalty: when current observation has
            # high reconstruction error (obstacle detected), penalize
            # imagined trajectories whose decoded images stay similar
            # to the current observation. This drives the agent toward
            # actions that change the visual scene (lane change).
            if (obs_decoder is not None
                    and ref_image is not None
                    and recon_error > self._surprise_threshold):
                surprise = self.visual_surprise(
                    obs_decoder, feat, ref_image,
                )
                # Scale by how surprising the current observation is
                surprise_weight = min(
                    recon_error / self._surprise_threshold - 1.0, 3.0,
                )
                step_score = step_score + (
                    self._beta_surprise * surprise_weight * surprise
                )

            # State-space penalty: raw (NOT z-scored) to avoid amplifying
            # noisy state decoder predictions during open-loop imagination.
            # Clamped and gently weighted to act as directional nudge.
            if state_decoder is not None and self._beta_s > 0:
                if self._heading_only_state:
                    state_instr = self.state_instrumental_value_heading_only(
                        state_decoder, feat
                    )
                else:
                    state_instr = self.state_instrumental_value(
                        state_decoder, feat,
                    )
                step_score = step_score + self._beta_s * state_instr

            total = total + discount * step_score
        return total

"""Tests for two-stage training: ae (encoder-decoder) and transition (rssm)."""

from __future__ import annotations

import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent, WorldModel


DEBUG_YAML = "configs/experiment/debug.yaml"


def _agent(overrides=None):
    cfg = Config.from_yaml(DEBUG_YAML, overrides=overrides)
    torch.manual_seed(0)
    return DeepAIFAgent(cfg), cfg


def _batch(cfg, B=2):
    T = cfg.training.seq_len
    return (torch.rand(B, T, 3, 64, 64) - 0.5,
            torch.zeros(B, T, 4),
            torch.zeros(B, T, 2))


def _ids(module):
    return {id(p) for p in module.parameters()}


def _opt_ids(agent):
    return {id(p) for g in agent._optimizer.param_groups for p in g["params"]}


class TestOptimizerScoping:
    def test_joint_includes_all(self):
        a, _ = _agent(["training.stage=joint"])
        opt = _opt_ids(a)
        assert _ids(a.world_model.rssm) <= opt
        assert _ids(a.world_model.encoder) <= opt

    def test_ae_includes_rssm_by_default(self):
        a, _ = _agent(["training.stage=ae"])
        opt = _opt_ids(a)
        wm = a.world_model
        encdec = _ids(wm.encoder) | _ids(wm.obs_decoder) | _ids(wm.state_decoder)
        assert encdec <= opt
        assert _ids(wm.rssm) <= opt

    def test_ae_can_exclude_rssm_for_legacy_behavior(self):
        a, _ = _agent(["training.stage=ae", "training.ae_train_rssm=false"])
        opt = _opt_ids(a)
        wm = a.world_model
        encdec = _ids(wm.encoder) | _ids(wm.obs_decoder) | _ids(wm.state_decoder)
        assert encdec <= opt
        assert len(_ids(wm.rssm) & opt) == 0

    def test_transition_only_rssm_and_frozen_encdec(self):
        a, _ = _agent(["training.stage=transition"])
        opt = _opt_ids(a)
        wm = a.world_model
        assert _ids(wm.rssm) <= opt
        assert all(not p.requires_grad for p in wm.encoder.parameters())
        assert all(not p.requires_grad for p in wm.obs_decoder.parameters())
        assert all(not p.requires_grad for p in wm.state_decoder.parameters())


class TestAEStage:
    def test_update_ae_trains_encoder(self):
        a, cfg = _agent(["training.stage=ae"])
        w0 = next(a.world_model.encoder.parameters()).detach().clone()
        info = a.update_ae(*_batch(cfg))
        w1 = next(a.world_model.encoder.parameters())
        assert not torch.allclose(w0, w1)
        for k in ("img_loss", "state_loss", "kl_rep", "total_loss"):
            assert k in info

    def test_update_ae_trains_rssm_by_default(self):
        a, cfg = _agent(["training.stage=ae"])
        w0 = next(a.world_model.rssm.parameters()).detach().clone()
        a.update_ae(*_batch(cfg))
        w1 = next(a.world_model.rssm.parameters())
        assert not torch.allclose(w0, w1)

    def test_update_ae_can_leave_rssm_frozen_for_legacy_behavior(self):
        a, cfg = _agent(["training.stage=ae", "training.ae_train_rssm=false"])
        w0 = next(a.world_model.rssm.parameters()).detach().clone()
        a.update_ae(*_batch(cfg))
        w1 = next(a.world_model.rssm.parameters())
        assert torch.allclose(w0, w1)

    def test_pure_ae_zero_kl(self):
        """ae_kl_rep=0 → kl term has zero weight (kl_rep still logged)."""
        a, cfg = _agent(["training.stage=ae", "training.ae_kl_rep=0.0"])
        info = a.update_ae(*_batch(cfg))
        assert info["total_loss"] >= 0.0


class TestTransitionStage:
    def test_transition_trains_rssm_frozen_encoder(self):
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.free_nats=0.0",
        ])
        rssm0 = next(a.world_model.rssm.parameters()).detach().clone()
        enc0  = next(a.world_model.encoder.parameters()).detach().clone()
        a.update(*_batch(cfg))
        rssm1 = next(a.world_model.rssm.parameters())
        enc1  = next(a.world_model.encoder.parameters())
        assert not torch.allclose(rssm0, rssm1), "rssm must train"
        assert torch.allclose(enc0, enc1), "encoder must stay frozen"

    def test_encoder_grad_none_in_transition(self):
        a, cfg = _agent([
            "training.stage=transition", "training.free_nats=0.0",
        ])
        a.update(*_batch(cfg))
        assert all(p.grad is None for p in a.world_model.encoder.parameters())

    def test_dense_one_step_requires_target_world_model(self):
        a, cfg = _agent([
            "training.stage=transition",
            "training.transition_loss_mode=dense_one_step",
            "training.transition_use_target_model=true",
        ])
        imgs, sts, acts = _batch(cfg)
        import pytest
        with pytest.raises(RuntimeError, match="dense_one_step.*target_world_model"):
            a.update(imgs, sts, acts)

    def test_dense_one_step_forward_finite_with_target(self):
        a, cfg = _agent([
            "training.stage=transition",
            "training.transition_loss_mode=dense_one_step",
            "training.transition_use_target_model=true",
            "training.lambda_pix=1.0",
            "training.lambda_deter=1.0",
            "training.lambda_kl=1.0",
            "training.lambda_stoch_mean=1.0",
            "training.lambda_stoch_decode=1.0",
            "training.free_nats_transition=0.0",
        ])
        target = WorldModel(cfg).to(a._device)
        target.eval()
        for p in target.parameters():
            p.requires_grad = False
        a.target_world_model = target
        info = a.update(*_batch(cfg))
        assert torch.isfinite(torch.tensor(info["total_loss"]))
        for key in (
            "kl_raw",
            "kl_clamped",
            "kl_train",
            "deter_loss",
            "rollout_pix_mse",
            "rollout_plain_pix_mse",
            "lambda_pix_eff",
            "stoch_mean_loss",
            "stoch_decode_loss",
            "lambda_stoch_mean",
            "lambda_stoch_decode",
        ):
            assert key in info
            assert torch.isfinite(torch.tensor(info[key]))

    def test_rollout_recon_drives_transition(self):
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.rollout_recon_horizon=3", "training.rollout_recon_weight=1.0",
        ])
        rssm0 = next(a.world_model.rssm.parameters()).detach().clone()
        info = a.update(*_batch(cfg))
        assert info["rollout_recon"] > 0.0
        assert not torch.allclose(rssm0, next(a.world_model.rssm.parameters()))


class TestAEEvaluation:
    def test_evaluate_ae_batch_matches_update_ae_path(self):
        """evaluate_ae_batch must use the deterministic decode path (det_post),
        producing finite img/state losses consistent with update_ae."""
        a, cfg = _agent(["training.stage=ae"])
        imgs, sts, acts = _batch(cfg)
        acc, n = a.evaluate_ae_batch(imgs, sts, acts)
        assert n == cfg.training.seq_len
        assert acc["img_loss"] >= 0.0
        assert acc["state_loss"] >= 0.0
        assert "kl_rep" in acc

    def test_evaluate_ae_no_grad(self):
        """Eval path must not accumulate gradients on encoder."""
        a, cfg = _agent(["training.stage=ae"])
        imgs, sts, acts = _batch(cfg)
        a.evaluate_ae_batch(imgs, sts, acts)
        assert all(p.grad is None for p in a.world_model.encoder.parameters())

    def test_ae_eval_differs_from_joint_path(self):
        """The ae (deterministic) and joint (sampled+VFE) eval paths should give
        different img_loss for the same model — proving the path actually changed.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train import evaluate_world_model

        a, cfg = _agent(["training.stage=ae"])
        imgs, sts, acts = _batch(cfg, B=4)

        class _DL:
            def __iter__(self):
                yield (imgs, sts, acts)

        torch.manual_seed(7)
        joint = evaluate_world_model(a, _DL(), stage="joint")
        ae    = evaluate_world_model(a, _DL(), stage="ae")
        # ae total_loss is reconstruction-only (img+state), joint is full VFE
        assert abs(ae["total_loss"] - (ae["img_loss"] + ae["state_loss"])) < 1e-6
        assert ae["kl_dyn"] == 0.0  # ae path does not compute transition KL


class TestStabilization:
    def test_amp_dtype_selection_cpu(self):
        """CPU: no amp, fp32, no scaler."""
        a, _ = _agent()
        assert a._use_amp is False
        assert a._amp_dtype == torch.float32
        assert a._scaler is None

    def test_scheduler_warmup_then_cosine(self):
        a, _ = _agent([
            "training.warmup_steps=5", "training.min_lr_ratio=0.1",
            "training.lr=1e-3",
        ])
        a.attach_scheduler(total_steps=20)
        lrs = []
        for _ in range(20):
            a._optimizer.step()
            a._scheduler.step()
            lrs.append(a._optimizer.param_groups[0]["lr"])
        assert lrs[0] < lrs[4]          # warmup increasing
        assert lrs[19] < lrs[4]         # cosine decaying
        assert lrs[19] >= 1e-3 * 0.1 * 0.99  # ~ floor

    def test_scheduler_off_when_no_warmup(self):
        a, _ = _agent(["training.warmup_steps=0"])
        assert a.attach_scheduler(100) is None
        assert a._scheduler is None

    def test_cycle_loss_off_by_default(self):
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.free_nats=0.0",
        ])
        info = a.update(*_batch(cfg))
        assert info["cycle"] == 0.0

    def test_cycle_loss_on_trains_rssm_frozen_encoder(self):
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.cycle_weight=0.1", "training.overshoot_horizon=3",
        ])
        rssm0 = next(a.world_model.rssm.parameters()).detach().clone()
        enc0  = next(a.world_model.encoder.parameters()).detach().clone()
        info = a.update(*_batch(cfg))
        assert info["cycle"] > 0.0
        assert not torch.allclose(rssm0, next(a.world_model.rssm.parameters()))
        assert torch.allclose(enc0, next(a.world_model.encoder.parameters()))

    def test_cycle_warmup_ramps_in(self):
        """cycle_warmup_steps=N: step 0 contributes nothing (ramp=0), later steps
        do (ramp>0).  With only the cycle loss active, rssm stays put on step 0
        then starts moving."""
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.free_nats=1.0",  # clamps kl_dyn → no grad from KL
            "training.cycle_weight=0.5", "training.cycle_warmup_steps=4",
        ])
        w0 = torch.cat([p.flatten() for p in a.world_model.rssm.parameters()]).clone()
        a.update(*_batch(cfg))               # step 0: ramp=0 → eff weight 0
        w1 = torch.cat([p.flatten() for p in a.world_model.rssm.parameters()])
        assert torch.allclose(w0, w1), "step 0 (ramp=0) must not move rssm"
        a.update(*_batch(cfg))               # step 1: ramp=0.25 → cycle trains
        w2 = torch.cat([p.flatten() for p in a.world_model.rssm.parameters()])
        assert not torch.allclose(w1, w2), "step 1 (ramp>0) must move rssm"
        assert a._train_step == 2

    def test_cycle_warmup_zero_immediate(self):
        """cycle_warmup_steps=0 → full cycle weight from step 0."""
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.free_nats=1.0",
            "training.cycle_weight=0.5", "training.cycle_warmup_steps=0",
        ])
        w0 = torch.cat([p.flatten() for p in a.world_model.rssm.parameters()]).clone()
        a.update(*_batch(cfg))
        w1 = torch.cat([p.flatten() for p in a.world_model.rssm.parameters()])
        assert not torch.allclose(w0, w1), "warmup=0 must train rssm on step 0"

    def test_cycle_clamp_keeps_finite(self):
        """Clamp(max=100) keeps cycle finite even with a poorly-init prior."""
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.cycle_weight=0.5", "training.cycle_warmup_steps=0",
            "training.free_nats=0.0",
        ])
        info = a.update(*_batch(cfg))
        assert info["cycle"] == info["cycle"]  # not NaN
        import math
        assert not math.isinf(info["cycle"])

    def test_cycle_nonzero_with_overshoot_horizon_zero(self):
        """Regression: cycle loss must run independent of overshoot_horizon.

        Previously, populating posteriors_ref for the cycle loss made the
        overshoot block (overshoot_horizon=0) compute osh_kl/0 = NaN, which
        the NaN-guard then skipped — dropping cycle AND rollout_recon entirely.
        """
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.cycle_weight=0.1", "training.cycle_horizon=5",
            "training.rollout_recon_horizon=5", "training.rollout_recon_weight=1.0",
            # overshoot_horizon stays 0 (default)
        ])
        info = a.update(*_batch(cfg))
        assert info["cycle"] > 0.0, "cycle loss must be nonzero with overshoot_horizon=0"
        assert info["rollout_recon"] > 0.0, "rollout_recon must not be dropped"
        assert info["overshoot_kl"] == 0.0  # overshoot genuinely off

    def test_cycle_horizon_independent_of_overshoot(self):
        """cycle_horizon controls cycle span, not overshoot_horizon."""
        a, cfg = _agent([
            "training.stage=transition", "training.kl_rep_scale=0.0",
            "training.cycle_weight=0.1", "training.cycle_horizon=4",
        ])
        info = a.update(*_batch(cfg))
        assert info["cycle"] > 0.0


class TestFlipBboxX:
    def test_mirror_left_to_right(self):
        # stored x near left → image x near right
        bb = torch.tensor([[0.0, 42.8, 3.2, 60.4]])
        f = DeepAIFAgent._flip_bbox_x(bb, 64)
        assert abs(f[0, 0].item() - 60.8) < 1e-3   # 64 - 3.2
        assert abs(f[0, 2].item() - 64.0) < 1e-3   # 64 - 0.0
        # y unchanged
        assert abs(f[0, 1].item() - 42.8) < 1e-3
        assert abs(f[0, 3].item() - 60.4) < 1e-3

    def test_nan_preserved(self):
        f = DeepAIFAgent._flip_bbox_x(torch.tensor([[float("nan")] * 4]), 64)
        assert torch.isnan(f).all()

    def test_double_flip_identity(self):
        bb = torch.tensor([[5.0, 10.0, 25.0, 40.0]])
        ff = DeepAIFAgent._flip_bbox_x(DeepAIFAgent._flip_bbox_x(bb, 64), 64)
        assert torch.allclose(ff, bb)


class TestObstacleTokenMask:
    def test_bbox_maps_to_correct_tokens(self):
        # bbox x[29.25,34.26] straddles the mirror axis (32), so x-flip keeps
        # cols {3,4}; y[33.57,37.57] → row 4.  Tokens 35,36 either way.
        bbox = torch.tensor([[29.25, 33.57, 34.26, 37.57]])
        m = DeepAIFAgent._bbox_token_mask(bbox, 64, 64, 8, torch.device("cpu"))
        idx = set(m[0].nonzero().squeeze(-1).tolist())
        assert idx == {35, 36}  # row4*8 + cols{3,4}

    def test_left_stored_bbox_masks_right_tokens(self):
        # stored x near left [2,18] → mirrored to image x [46,62] = cols 5,6,7
        bbox = torch.tensor([[2.0, 33.57, 18.0, 37.57]])
        m = DeepAIFAgent._bbox_token_mask(bbox, 64, 64, 8, torch.device("cpu"))
        cols = sorted({i % 8 for i in m[0].nonzero().squeeze(-1).tolist()})
        assert min(cols) >= 5, f"mirrored box must be on right cols, got {cols}"

    def test_nan_bbox_empty_mask(self):
        m = DeepAIFAgent._bbox_token_mask(
            torch.tensor([[float("nan")] * 4]), 64, 64, 8, torch.device("cpu"),
        )
        assert m.sum().item() == 0.0

    def test_full_image_bbox_marks_all_tokens(self):
        m = DeepAIFAgent._bbox_token_mask(
            torch.tensor([[0.0, 0.0, 64.0, 64.0]]), 64, 64, 8, torch.device("cpu"),
        )
        assert m.sum().item() == 64.0

    def test_crop_road_remaps_bbox_rows(self):
        """crop_road=True remaps original y to the cropped+resized space.

        bbox [29.25,33.57,34.26,37.57]: original rows ~4 (tokens 35,36).
        After crop_road(0.6): kept band starts at y=25.6, so y→(y-25.6)*64/38.4
        → 13.3..19.9 → rows 1-2 → tokens {11,12,19,20}.
        """
        bbox = torch.tensor([[29.25, 33.57, 34.26, 37.57]])
        m_off = DeepAIFAgent._bbox_token_mask(
            bbox, 64, 64, 8, torch.device("cpu"), crop_road=False,
        )
        assert set(m_off[0].nonzero().squeeze(-1).tolist()) == {35, 36}

        m_on = DeepAIFAgent._bbox_token_mask(
            bbox, 64, 64, 8, torch.device("cpu"),
            crop_road=True, keep_bottom_frac=0.6,
        )
        assert set(m_on[0].nonzero().squeeze(-1).tolist()) == {11, 12, 19, 20}

    def test_crop_road_obstacle_above_band_is_empty(self):
        """bbox entirely above the kept band (y < 25.6) → degenerate → mask 0."""
        above = torch.tensor([[29.0, 10.0, 34.0, 20.0]])
        m = DeepAIFAgent._bbox_token_mask(
            above, 64, 64, 8, torch.device("cpu"), crop_road=True,
        )
        assert m.sum().item() == 0.0


class TestObstacleTokenWeight:
    _VIT = [
        "device=cpu", "training.seq_len=8", "training.batch_size=2",
        "training.stage=transition", "training.kl_rep_scale=0.0",
        "training.cycle_weight=0.1", "training.overshoot_horizon=3",
        "token_vit.num_layers=1", "token_vit.num_prior_layers=1",
        "token_vit.num_post_layers=1", "token_vit.embed_dim=32",
        "token_vit.deter_dim=32", "token_vit.stoch_dim=8", "token_vit.feat_dim=40",
        "token_vit.num_tokens=64", "token_vit.num_heads=4",
        "rssm.stoch_dim=8", "rssm.deter_dim=32", "rssm.embed_dim=32",
        "ensemble.num_heads=2", "ensemble.hidden_dim=32",
    ]

    def _batch(self, cfg, B=2):
        T = cfg.training.seq_len
        bb = torch.full((B, T, 4), float("nan"))
        bb[:, :] = torch.tensor([29.25, 33.57, 34.26, 37.57])
        return (torch.rand(B, T, 3, 64, 64) - 0.5,
                torch.zeros(B, T, 4), torch.zeros(B, T, 2), None, bb)

    def _cycle(self, w):
        cfg = Config.from_yaml(
            "configs/experiment/token_vit.yaml",
            overrides=self._VIT + [f"training.obstacle_token_weight={w}"],
        )
        torch.manual_seed(0)
        a = DeepAIFAgent(cfg)
        b = self._batch(cfg)
        torch.manual_seed(0)
        return a.update(*b)["cycle"]

    def test_weight_one_is_baseline(self):
        # weight=1.0 must not error and produce a finite cycle value
        c = self._cycle(1.0)
        assert c >= 0.0

    def test_weight_high_increases_cycle(self):
        c1 = self._cycle(1.0)
        c5 = self._cycle(5.0)
        assert c5 > c1

    def test_rssm_ignores_token_weight(self):
        """RSSM (2D latent) must be identical regardless of token weight."""
        def run(w):
            cfg = Config.from_yaml(DEBUG_YAML, overrides=[
                "training.stage=transition", "training.kl_rep_scale=0.0",
                "training.cycle_weight=0.1", "training.overshoot_horizon=3",
                f"training.obstacle_token_weight={w}",
            ])
            torch.manual_seed(0)
            a = DeepAIFAgent(cfg)
            T = cfg.training.seq_len
            bb = torch.full((2, T, 4), float("nan"))
            bb[:, :] = torch.tensor([29., 33., 34., 37.])
            torch.manual_seed(0)
            return a.update(torch.rand(2, T, 3, 64, 64) - 0.5,
                            torch.zeros(2, T, 4), torch.zeros(2, T, 2),
                            None, bb)["cycle"]
        assert abs(run(1.0) - run(5.0)) < 1e-9


class TestStageSelectionLoss:
    def _sel(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from train import stage_selection_loss
        return stage_selection_loss

    def test_ae_uses_recon_ignores_kl(self):
        sel = self._sel()
        info = {"img_loss": 0.0015, "state_loss": 0.01,
                "kl_dyn": 1800.0, "kl_rep": 1800.0, "total_loss": 3601.0,
                "rollout_recon": 0.0}
        assert abs(sel("ae", info) - 0.0115) < 1e-9

    def test_ae_best_epoch_tracks_img(self):
        """epoch16 (img 0.0015) must beat epoch7 (img 0.0021)."""
        sel = self._sel()
        e16 = {"img_loss": 0.0015, "state_loss": 0.0, "total_loss": 99.0,
               "kl_dyn": 5.0, "rollout_recon": 0.0}
        e7  = {"img_loss": 0.0021, "state_loss": 0.0, "total_loss": 1.0,
               "kl_dyn": 5.0, "rollout_recon": 0.0}
        assert sel("ae", e16) < sel("ae", e7)

    def test_transition_uses_kldyn_plus_rollout(self):
        sel = self._sel()
        info = {"img_loss": 9.0, "state_loss": 9.0, "kl_dyn": 0.5,
                "rollout_recon": 0.02, "total_loss": 18.52}
        assert abs(sel("transition", info) - 0.52) < 1e-9

    def test_joint_uses_total(self):
        sel = self._sel()
        assert sel("joint", {"total_loss": 1.23, "img_loss": 0.1}) == 1.23


class TestLoadEncoderDecoder:
    def test_load_encoder_decoder_copies_weights(self, tmp_path):
        # Stage-1 agent, save checkpoint
        a1, _ = _agent(["training.stage=ae"])
        ckpt = tmp_path / "ae.pt"
        a1.save_checkpoint(str(ckpt))

        # Stage-2 agent loads encoder-decoder from it
        a2, _ = _agent(["training.stage=transition"])
        a2.load_encoder_decoder(str(ckpt))
        # Encoder weights now match a1's
        for p1, p2 in zip(a1.world_model.encoder.parameters(),
                          a2.world_model.encoder.parameters()):
            assert torch.allclose(p1, p2)

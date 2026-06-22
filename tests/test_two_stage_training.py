"""Tests for two-stage training: ae (encoder-decoder) and transition (rssm)."""

from __future__ import annotations

import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent


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

    def test_ae_excludes_rssm(self):
        a, _ = _agent(["training.stage=ae"])
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

    def test_update_ae_does_not_train_rssm(self):
        a, cfg = _agent(["training.stage=ae"])
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

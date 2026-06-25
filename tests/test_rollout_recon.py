"""Tests for multi-step pixel rollout reconstruction loss and diagnostic grid."""

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
    imgs = torch.rand(B, T, 3, 64, 64) - 0.5
    sts  = torch.zeros(B, T, 4)
    acts = torch.zeros(B, T, 2)
    acts[:, :, 1] = 0.3
    return imgs, sts, acts


class TestRolloutReconLoss:
    def test_off_by_default(self):
        agent, cfg = _agent()
        info = agent.update(*_batch(cfg))
        assert info["rollout_recon"] == 0.0

    def test_on_produces_positive_loss(self):
        agent, cfg = _agent([
            "training.rollout_recon_horizon=3",
            "training.rollout_recon_weight=1.0",
        ])
        info = agent.update(*_batch(cfg))
        assert info["rollout_recon"] > 0.0
        assert not torch.isnan(torch.tensor(info["total_loss"]))

    def test_increases_total_loss(self):
        """With same seed, enabling rollout recon changes total loss."""
        a_off, cfg = _agent()
        batch = _batch(cfg)
        i_off = a_off.update(*[b.clone() for b in batch])

        a_on, cfg2 = _agent([
            "training.rollout_recon_horizon=3",
            "training.rollout_recon_weight=1.0",
        ])
        a_on.world_model.load_state_dict(a_off.world_model.state_dict())
        i_on = a_on.update(*[b.clone() for b in batch])
        assert i_on["total_loss"] > i_off["total_loss"]

    def test_bbox_weighting_runs(self):
        agent, cfg = _agent([
            "training.rollout_recon_horizon=3",
            "training.rollout_recon_weight=1.0",
            "training.rollout_recon_obstacle_weight=5.0",
        ])
        imgs, sts, acts = _batch(cfg)
        T = cfg.training.seq_len
        bbox = torch.tensor([[[10., 10., 40., 40.]] * T] * 2)  # (B,T,4)
        info = agent.update(imgs, sts, acts, obstacle_bbox=bbox)
        assert info["rollout_recon"] > 0.0

    def test_step_decay_changes_loss(self):
        batch = None
        a1, cfg1 = _agent([
            "training.rollout_recon_horizon=4",
            "training.rollout_recon_weight=1.0",
            "training.rollout_recon_step_decay=0.0",
        ])
        batch = _batch(cfg1)
        i1 = a1.update(*[b.clone() for b in batch])

        a2, cfg2 = _agent([
            "training.rollout_recon_horizon=4",
            "training.rollout_recon_weight=1.0",
            "training.rollout_recon_step_decay=0.5",
        ])
        a2.world_model.load_state_dict(a1.world_model.state_dict())
        i2 = a2.update(*[b.clone() for b in batch])
        # Decay reweights steps → recon value differs
        assert abs(i1["rollout_recon"] - i2["rollout_recon"]) > 1e-6


class TestImgReconObstacleWeight:
    def _bbox(self, cfg, B=2):
        T = cfg.training.seq_len
        return torch.tensor([[[10., 10., 40., 40.]] * T] * B)

    def test_weight_one_ignores_bbox(self):
        """img_recon_obstacle_weight=1.0 → bbox has no effect (regression)."""
        agent, cfg = _agent()
        agent2, cfg2 = _agent()
        # Sync weights BEFORE any update (update mutates weights via optimizer).
        agent2.world_model.load_state_dict(agent.world_model.state_dict())
        batch = _batch(cfg)

        torch.manual_seed(123)
        i_nobbox = agent.update(*[b.clone() for b in batch])
        torch.manual_seed(123)
        i_bbox = agent2.update(*[b.clone() for b in batch],
                               obstacle_bbox=self._bbox(cfg2))
        assert abs(i_nobbox["img_loss"] - i_bbox["img_loss"]) < 1e-9

    def test_weight_high_with_bbox_changes_img_loss(self):
        a_base, cfg = _agent()
        a_w, cfg_w = _agent(["training.img_recon_obstacle_weight=5.0"])
        a_w.world_model.load_state_dict(a_base.world_model.state_dict())
        batch = _batch(cfg)

        torch.manual_seed(123)
        i_base = a_base.update(*[b.clone() for b in batch],
                               obstacle_bbox=self._bbox(cfg))
        torch.manual_seed(123)
        i_w = a_w.update(*[b.clone() for b in batch],
                         obstacle_bbox=self._bbox(cfg_w))
        assert i_w["img_loss"] > i_base["img_loss"]

    def test_weight_high_without_bbox_is_regression(self):
        """weight>1 but bbox=None → identical to baseline."""
        a_base, cfg = _agent()
        a_w, cfg_w = _agent(["training.img_recon_obstacle_weight=5.0"])
        a_w.world_model.load_state_dict(a_base.world_model.state_dict())
        batch = _batch(cfg)

        torch.manual_seed(123)
        i_base = a_base.update(*[b.clone() for b in batch])
        torch.manual_seed(123)
        i_w = a_w.update(*[b.clone() for b in batch])  # no bbox
        assert abs(i_base["img_loss"] - i_w["img_loss"]) < 1e-9


class TestBboxWeightMap:
    def test_uniform_outside_bbox(self):
        # _bbox_weight_map applies a CARLA x-mirror: stored x [10,30] maps to
        # image columns [64-30, 64-10] = [34, 54]; y is unchanged [10,30].
        bbox = torch.tensor([[10., 10., 30., 30.]])
        wm = DeepAIFAgent._bbox_weight_map(
            bbox, (1, 3, 64, 64), 5.0, torch.device("cpu"),
        )
        assert wm.shape == (1, 1, 64, 64)
        assert wm[0, 0, 0, 0].item() == 1.0          # corner outside bbox
        assert wm[0, 0, 20, 20].item() == 1.0        # left of mirrored box → outside
        assert wm[0, 0, 20, 44].item() == 5.0        # (row20,col44) inside mirrored box

    def test_nan_bbox_gives_uniform(self):
        bbox = torch.tensor([[float("nan")] * 4])
        wm = DeepAIFAgent._bbox_weight_map(
            bbox, (1, 3, 64, 64), 5.0, torch.device("cpu"),
        )
        assert torch.allclose(wm, torch.ones_like(wm))


class TestDiagGrid:
    def test_make_diag_grid_creates_file(self, tmp_path):
        from active_inference.training.diag_grid import make_diag_grid
        cfg = Config.from_yaml(DEBUG_YAML)
        torch.manual_seed(0)
        agent = DeepAIFAgent(cfg)

        T = 20
        imgs = torch.rand(T, 3, 64, 64) - 0.5
        sts  = torch.zeros(T, 4)
        acts = torch.zeros(T, 2); acts[:, 1] = 0.3

        out = tmp_path / "diag.png"
        ok = make_diag_grid(
            agent.world_model, imgs, sts, acts,
            start_idx=4, output_path=str(out),
            horizons=[1, 5, 10, 15], context_len=5,
            device=agent._device, title="test",
        )
        assert ok is True
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_false_when_no_future_frames(self, tmp_path):
        from active_inference.training.diag_grid import make_diag_grid
        cfg = Config.from_yaml(DEBUG_YAML)
        agent = DeepAIFAgent(cfg)
        T = 5
        imgs = torch.rand(T, 3, 64, 64) - 0.5
        sts  = torch.zeros(T, 4)
        acts = torch.zeros(T, 2)
        # start_idx near the end → no horizons fit
        ok = make_diag_grid(
            agent.world_model, imgs, sts, acts,
            start_idx=4, output_path=str(tmp_path / "x.png"),
            horizons=[5, 10], device=agent._device,
        )
        assert ok is False

import torch
from active_inference.config import Config
from active_inference.agent import DeepAIFAgent


def _debug_cfg():
    return Config.from_yaml("configs/experiment/debug.yaml")


def test_forward_pass():
    agent = DeepAIFAgent(_debug_cfg())
    agent.reset()
    img = torch.randn(3, 64, 64)
    state = torch.randn(4)
    action = agent.step(img, state)
    assert action.shape == (2,)


def test_action_range():
    agent = DeepAIFAgent(_debug_cfg())
    agent.reset()
    action = agent.step(torch.randn(3, 64, 64), torch.randn(4))
    assert (action >= -1.0).all() and (action <= 1.0).all()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(42)
    cfg = _debug_cfg()
    agent = DeepAIFAgent(cfg)
    agent.reset()
    img = torch.randn(3, 64, 64)
    st = torch.randn(4)

    path = str(tmp_path / "ckpt.pt")
    agent.save_checkpoint(path)

    agent2 = DeepAIFAgent(cfg)
    agent2.load_checkpoint(path)

    p1 = dict(agent.world_model.named_parameters())
    p2 = dict(agent2.world_model.named_parameters())
    for k in p1:
        assert torch.allclose(p1[k], p2[k]), f"Mismatch at {k}"


def test_update_reduces_loss():
    torch.manual_seed(0)
    cfg = _debug_cfg()
    agent = DeepAIFAgent(cfg)
    B, T = 2, cfg.training.seq_len
    images = torch.randn(B, T, 3, 64, 64)
    states = torch.randn(B, T, 4)
    actions = torch.randn(B, T, 2).clamp(-1, 1)

    info1 = agent.update(images, states, actions)
    for _ in range(4):
        info2 = agent.update(images, states, actions)
    assert info2["total_loss"] < info1["total_loss"]


def test_action_to_carla():
    # Linear remap: accel ∈ [-1,1] → throttle ∈ [0.35, 0.55]
    # Formula: throttle = 0.35 + 0.2 * (accel + 1) / 2

    # accel=0.7 → 0.35 + 0.2 * 1.7/2 = 0.52
    action = torch.tensor([0.5, 0.7])
    s, t, b = DeepAIFAgent.action_to_carla(action)
    assert abs(s - 0.5) < 1e-5
    assert abs(t - 0.52) < 1e-5
    assert abs(b - 0.0) < 1e-5

    # accel=-1.0 → min_throttle=0.35
    action_neg = torch.tensor([-0.3, -1.0])
    s, t, b = DeepAIFAgent.action_to_carla(action_neg)
    assert abs(t - 0.35) < 1e-5
    assert abs(b - 0.0) < 1e-5

    # accel=0.0 → midpoint=0.45
    action_zero = torch.tensor([0.0, 0.0])
    s, t, b = DeepAIFAgent.action_to_carla(action_zero)
    assert abs(t - 0.45) < 1e-5
    assert abs(b - 0.0) < 1e-5

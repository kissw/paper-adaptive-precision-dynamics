"""Integration test: train on synthetic 4D data and verify steering emerges.

Uses debug config (small model) for fast CPU execution.
Tests the full pipeline: data -> train -> preference -> plan -> verify steering.
"""
import numpy as np
import pytest
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.data.synthetic import SyntheticDrivingData
from active_inference.data.dataset import get_dataloader, get_preference_dataloader
from active_inference.utils.seed import set_seed


@pytest.fixture(scope="module")
def trained_agent(tmp_path_factory):
    """Train a small agent on synthetic data (shared across tests in this module)."""
    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    cfg.training.epochs = 3
    cfg.preference.warmup_epoch = 1
    set_seed(cfg.seed)

    # Generate synthetic data
    synth = SyntheticDrivingData()
    data = synth.generate(n_episodes=10, episode_len=cfg.training.seq_len * 4)
    data_path = str(tmp_path_factory.mktemp("data") / "integration.h5")
    synth.to_hdf5(data, data_path)

    # Train world model
    agent = DeepAIFAgent(cfg)
    dataloader = get_dataloader(
        data_path,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=0,
        shuffle=True,
    )

    losses = []
    for epoch in range(3):
        epoch_losses = []
        for images, states, actions in dataloader:
            info = agent.update(images, states, actions)
            if not np.isnan(info["total_loss"]):
                epoch_losses.append(info["total_loss"])
        losses.append(np.mean(epoch_losses) if epoch_losses else float("nan"))

    # Update preference model
    pref_loader = get_preference_dataloader(
        data_path,
        batch_size=8,
        seq_len=cfg.training.seq_len,
        success_only=True,
        num_workers=0,
    )
    if len(pref_loader.dataset) > 0:
        agent.update_preference(pref_loader, max_samples=cfg.preference.max_samples)

    return agent, losses, cfg


def test_training_loss_decreases(trained_agent):
    """Training loss should decrease over 3 epochs."""
    _, losses, _ = trained_agent
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_no_shape_mismatch(trained_agent):
    """No shape errors during forward pass with 4D state."""
    agent, _, _ = trained_agent
    agent.reset()
    img = torch.randn(3, 64, 64)
    state = torch.tensor([30.0, 0.0, 0.1, -0.1])
    action = agent.step(img, state)
    assert action.shape == (2,)


def test_cem_varied_steering(trained_agent):
    """CEM should produce varied steering across different scenarios."""
    agent, _, _ = trained_agent
    steers = []
    scenarios = [
        [30.0, 0.0, 0.0, 0.0],    # centered
        [30.0, 0.0, 0.3, 0.2],    # heading left + offset left
        [30.0, 0.0, -0.3, -0.2],  # heading right + offset right
        [30.0, 0.0, 0.0, 0.4],    # offset left only
        [30.0, 0.0, 0.0, -0.4],   # offset right only
        [30.0, 0.0, 0.5, 0.0],    # heading left only
        [30.0, 0.0, -0.5, 0.0],   # heading right only
        [30.0, 0.3, 0.2, 0.1],    # already steering left
        [30.0, -0.3, -0.2, -0.1], # already steering right
        [30.0, 0.0, 0.4, -0.3],   # heading left, offset right
    ]
    for state_vals in scenarios:
        agent.reset()
        img = torch.randn(3, 64, 64) * 0.1
        state = torch.tensor(state_vals)
        plan = agent.step_with_info(img, state)
        steers.append(float(plan.action[0]))

    steer_std = np.std(steers)
    assert steer_std > 0.01, f"Steering std too low: {steer_std:.4f}"


def test_integration_no_classical_control():
    """Verify no classical controllers exist in the planning/training code."""
    import importlib
    import inspect

    modules = [
        "active_inference.planning.cem_planner",
        "active_inference.planning.efe",
        "active_inference.training.losses",
        "active_inference.training.preference",
        "active_inference.agent",
    ]
    banned = ["stanley", "pid_control", "k_heading", "k_crosstrack"]

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        source = inspect.getsource(mod)
        source_lower = source.lower()
        for term in banned:
            assert term not in source_lower, (
                f"Found banned term '{term}' in {mod_name}"
            )

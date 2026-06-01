import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch

from active_inference.training.preference import PreferenceModel

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_gmm_log_prob_higher_at_means():
    pref = PreferenceModel(K=3, latent_dim=8)
    with torch.no_grad():
        at_means = pref.log_prob(pref.means)
        at_random = pref.log_prob(torch.randn(3, 8) * 10)
    assert (at_means > at_random).all()


def test_gmm_fit_recovers_known():
    torch.manual_seed(0)
    # Create data from known 2-component mixture in 8 dims
    true_means = torch.tensor([[2.0] * 8, [-2.0] * 8])
    data = torch.cat(
        [
            true_means[0].unsqueeze(0) + torch.randn(500, 8) * 0.3,
            true_means[1].unsqueeze(0) + torch.randn(500, 8) * 0.3,
        ]
    )
    pref = PreferenceModel(K=2, latent_dim=8)
    pref.update_from_latents(data, n_iters=200, lr=0.01)
    fitted_means = pref.means.data.sort(dim=0).values
    true_sorted = true_means.sort(dim=0).values
    assert torch.allclose(fitted_means, true_sorted, atol=1.0)


def test_gmm_save_load_roundtrip(tmp_path):
    pref = PreferenceModel(K=3, latent_dim=16)
    z = torch.randn(10, 16)
    lp_before = pref.log_prob(z)
    path = str(tmp_path / "gmm.pt")
    pref.save_state(path)
    pref2 = PreferenceModel(K=3, latent_dim=16)
    pref2.load_state(path)
    lp_after = pref2.log_prob(z)
    assert torch.allclose(lp_before, lp_after)


def test_gmm_no_component_collapse():
    torch.manual_seed(42)
    data = torch.cat(
        [
            torch.randn(300, 8) + 3,
            torch.randn(300, 8) - 3,
            torch.randn(300, 8),
        ]
    )
    pref = PreferenceModel(K=3, latent_dim=8)
    pref.update_from_latents(data, n_iters=200)
    # All K means should be distinct
    means = pref.means.data
    for i in range(3):
        for j in range(i + 1, 3):
            dist = (means[i] - means[j]).norm()
            assert dist > 0.5, f"Components {i} and {j} collapsed: dist={dist:.3f}"


def test_gmm_distribution_shape():
    pref = PreferenceModel(K=3, latent_dim=64)
    z = torch.randn(10, 64)
    lp = pref.log_prob(z)
    assert lp.shape == (10,)


# ─────────────────────────────────────────────────────────────────────────────
# encode_latents regression tests (Step E)
# ─────────────────────────────────────────────────────────────────────────────

def _make_h5_data(tmp_path, n_episodes: int = 3, frames_per_ep: int = 5,
                  stoch_dim: int = 8) -> str:
    """Write a synthetic HDF5 with two episodes for encode_latents tests."""
    T = n_episodes * frames_per_ep
    path = str(tmp_path / "data.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=np.zeros((T, 3, 64, 64), dtype=np.float32))
        f.create_dataset("states", data=np.zeros((T, 4), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros((T, 2), dtype=np.float32))
        ep_ids = np.repeat(np.arange(n_episodes), frames_per_ep).astype(np.int64)
        f.create_dataset("episode_ids", data=ep_ids)
    return path


def _make_mock_agent(stoch_dim: int = 8, action_dim: int = 2):
    """Build a mock agent that tracks RSSM state resets."""
    from active_inference.config import Config
    from active_inference.agent import DeepAIFAgent

    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    cfg.rssm.stoch_dim = stoch_dim
    cfg.cem.action_dim = action_dim
    agent = DeepAIFAgent(cfg)
    return agent


def test_encode_latents_resets_state_at_episode_boundary(tmp_path):
    """State must be reset when episode_ids[i] != episode_ids[i-1]."""
    from fit_contrastive_preference import encode_latents
    from active_inference.models.rssm import RSSMState

    agent = _make_mock_agent()
    data_path = _make_h5_data(tmp_path, n_episodes=2, frames_per_ep=4)

    reset_calls = []
    original_initial = agent.world_model.rssm.initial

    def tracking_initial(batch_size, device):
        reset_calls.append(1)
        return original_initial(batch_size, device)

    agent.world_model.rssm.initial = tracking_initial
    encode_latents(agent, data_path, max_samples=8)

    # initial() called once at start + once at each episode boundary
    # 2 episodes → 2 calls (one at i=0, one at the boundary between ep 0 and ep 1)
    assert len(reset_calls) >= 2, (
        f"Expected >= 2 initial() calls (one per episode), got {len(reset_calls)}"
    )


def test_encode_latents_prev_act_uses_previous_action(tmp_path):
    """prev_act at step i should equal actions[i-1]."""
    from fit_contrastive_preference import encode_latents

    agent = _make_mock_agent()
    n_frames = 5
    data_path = str(tmp_path / "single.h5")

    # Distinct action pattern so we can detect which action is used
    actions = np.zeros((n_frames, 2), dtype=np.float32)
    actions[2] = [0.5, 0.5]  # frame 2 has a non-zero action

    with h5py.File(data_path, "w") as f:
        f.create_dataset("images", data=np.zeros((n_frames, 3, 64, 64), dtype=np.float32))
        f.create_dataset("states", data=np.zeros((n_frames, 4), dtype=np.float32))
        f.create_dataset("actions", data=actions)
        f.create_dataset("episode_ids", data=np.zeros(n_frames, dtype=np.int64))

    prev_acts_seen = []
    original_obs_step = agent.world_model.rssm.obs_step

    def tracking_obs_step(prev_state, prev_act, embed):
        prev_acts_seen.append(prev_act.detach().cpu().clone())
        return original_obs_step(prev_state, prev_act, embed)

    agent.world_model.rssm.obs_step = tracking_obs_step
    encode_latents(agent, data_path, max_samples=n_frames)

    # At step i=3, prev_act should equal actions[2] = [0.5, 0.5]
    assert len(prev_acts_seen) == n_frames
    act_at_step3 = prev_acts_seen[3]
    assert torch.allclose(act_at_step3, torch.tensor([[0.5, 0.5]]), atol=1e-5), (
        f"Expected prev_act at i=3 to be actions[2]=[0.5,0.5], got {act_at_step3}"
    )


def test_encode_latents_single_episode_matches_sequential_rollout(tmp_path):
    """Single episode: encode_latents returns deterministic posterior means.

    We verify by running encode_latents twice on identical data with the same
    random seed and confirming both produce identical latents (i.e., the
    post.mean path is deterministic given fixed model weights and inputs).
    We also verify the first frame mean matches a fresh manual computation.
    """
    from fit_contrastive_preference import encode_latents

    agent = _make_mock_agent()
    n = 6
    data_path = str(tmp_path / "one_ep.h5")

    np.random.seed(0)
    actions_np = np.random.randn(n, 2).astype(np.float32)
    images_np = np.zeros((n, 3, 64, 64), dtype=np.float32)  # zeros for determinism
    states_np = np.zeros((n, 4), dtype=np.float32)

    with h5py.File(data_path, "w") as f:
        f.create_dataset("images", data=images_np)
        f.create_dataset("states", data=states_np)
        f.create_dataset("actions", data=actions_np)
        f.create_dataset("episode_ids", data=np.zeros(n, dtype=np.int64))

    # Run twice with same seed — should produce identical post.mean values
    torch.manual_seed(42)
    latents_a = encode_latents(agent, data_path, max_samples=n)
    torch.manual_seed(42)
    latents_b = encode_latents(agent, data_path, max_samples=n)

    assert torch.allclose(latents_a, latents_b, atol=1e-5), (
        "Two identical encode_latents runs with same seed must produce same means"
    )

    # Frame 0: manual first-step computation (same seed, zero inputs)
    dev = agent._device
    wm = agent.world_model
    torch.manual_seed(42)
    with torch.no_grad():
        img0 = torch.zeros(1, 3, 64, 64, device=dev)
        st0 = torch.zeros(1, 4, device=dev)
        state0 = wm.rssm.initial(1, dev)
        prev_act0 = torch.zeros(1, 2, device=dev)
        embed0 = wm.encode_obs(img0, st0)
        post0, _ = wm.rssm.obs_step(state0, prev_act0, embed0)
        manual_mean0 = post0.mean.cpu()

    assert torch.allclose(latents_a[0:1].cpu(), manual_mean0, atol=1e-5), (
        "First latent from encode_latents must match direct obs_step output"
    )

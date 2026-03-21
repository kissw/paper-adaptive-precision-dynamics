"""Verify that the Active Inference pipeline learns meaningful steering.

Runs a short training loop with synthetic 4D data, then tests CEM planning
to confirm steering emerges from the world model + EFE framework.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent
from active_inference.data.synthetic import SyntheticDrivingData
from active_inference.data.dataset import get_dataloader, get_preference_dataloader
from active_inference.utils.seed import set_seed


def generate_data(cfg, output_dir: Path) -> str:
    """Generate synthetic 4D driving data and save to HDF5."""
    synth = SyntheticDrivingData()
    data = synth.generate(n_episodes=10, episode_len=cfg.training.seq_len * 4)
    data_path = str(output_dir / "verify_synthetic.h5")
    synth.to_hdf5(data, data_path)
    print(f"Generated synthetic data: {data['states'].shape[0]} frames, "
          f"state_dim={data['states'].shape[1]}")
    print(f"  Success ratio: {data['success_flags'].mean():.2f}")
    return data_path


def train_world_model(cfg, agent, data_path, n_epochs=5):
    """Train world model for n_epochs and return loss history."""
    dataloader = get_dataloader(
        data_path,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        num_workers=0,
        shuffle=True,
    )

    losses = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for images, states, actions in dataloader:
            info = agent.update(images, states, actions)
            if not np.isnan(info["total_loss"]):
                epoch_losses.append(info["total_loss"])

        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        losses.append(mean_loss)
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={mean_loss:.4f} "
              f"(img={info['img_loss']:.4f} state={info['state_loss']:.4f} "
              f"kl_dyn={info['kl_dyn']:.4f})")

    return losses


def update_preference_model(cfg, agent, data_path):
    """Fit preference GMM on successful trajectories."""
    pref_loader = get_preference_dataloader(
        data_path,
        batch_size=8,
        seq_len=cfg.training.seq_len,
        success_only=True,
        num_workers=0,
    )
    if len(pref_loader.dataset) == 0:
        print("  WARNING: No valid preference sequences found!")
        return {}

    pref_info = agent.update_preference(pref_loader, max_samples=cfg.preference.max_samples)
    print(f"  Preference GMM: log_prob={pref_info['pref_log_prob']:.2f} "
          f"min_dist={pref_info['pref_min_component_dist']:.2f} "
          f"samples={pref_info['pref_n_samples']}")
    return pref_info


def verify_cem_steering(cfg, agent):
    """Test CEM planning from various initial states and check steering varies."""
    agent.reset()

    # Create diverse initial observations
    scenarios = {
        "straight_centered": {"heading": 0.0, "crosstrack": 0.0},
        "left_curve": {"heading": 0.3, "crosstrack": 0.2},
        "right_curve": {"heading": -0.3, "crosstrack": -0.2},
        "drifting_left": {"heading": 0.1, "crosstrack": 0.4},
        "drifting_right": {"heading": -0.1, "crosstrack": -0.4},
        "strong_left": {"heading": 0.5, "crosstrack": 0.3},
        "strong_right": {"heading": -0.5, "crosstrack": -0.3},
        "centered_heading_left": {"heading": 0.4, "crosstrack": 0.0},
        "centered_heading_right": {"heading": -0.4, "crosstrack": 0.0},
        "offset_no_heading": {"heading": 0.0, "crosstrack": 0.3},
    }

    results = {}
    for name, params in scenarios.items():
        agent.reset()
        img = torch.randn(3, 64, 64) * 0.1
        state = torch.tensor([
            30.0,  # speed
            0.0,   # steer (current)
            params["heading"],
            params["crosstrack"],
        ])
        plan = agent.step_with_info(img, state)
        results[name] = {
            "steer": float(plan.action[0]),
            "accel": float(plan.action[1]),
            "efe": plan.efe_score,
            "epistemic": plan.epistemic_score,
        }

    # Print results table
    print("\n  CEM Planning Results:")
    print(f"  {'Scenario':<30} {'Steer':>8} {'Accel':>8} {'EFE':>10} {'Epist':>10}")
    print("  " + "-" * 70)
    for name, r in results.items():
        print(f"  {name:<30} {r['steer']:>8.3f} {r['accel']:>8.3f} "
              f"{r['efe']:>10.3f} {r['epistemic']:>10.3f}")

    return results


def check_metrics(losses, cem_results):
    """Check all acceptance criteria and return pass/fail."""
    all_pass = True

    # Story 2: Training loss decreases
    print("\n=== Story 2: Training Loop Verification ===")
    loss_decreases = losses[-1] < losses[0]
    print(f"  Loss decreases: {losses[0]:.4f} -> {losses[-1]:.4f} ... "
          f"{'PASS' if loss_decreases else 'FAIL'}")
    if not loss_decreases:
        all_pass = False

    # Story 3: CEM produces non-trivial steering
    print("\n=== Story 3: CEM Steering Verification ===")

    steers = [r["steer"] for r in cem_results.values()]
    steer_std = np.std(steers)
    steer_varied = steer_std > 0.05
    print(f"  Steering std across scenarios: {steer_std:.4f} ... "
          f"{'PASS' if steer_varied else 'FAIL (need > 0.05)'}")
    if not steer_varied:
        all_pass = False

    # Left vs right curve steering difference
    left_steers = [cem_results[k]["steer"] for k in
                   ["left_curve", "drifting_left", "strong_left"]]
    right_steers = [cem_results[k]["steer"] for k in
                    ["right_curve", "drifting_right", "strong_right"]]
    steer_diff = abs(np.mean(left_steers) - np.mean(right_steers))
    steer_diff_ok = steer_diff > 0.1
    print(f"  Left vs right mean steer diff: {steer_diff:.4f} ... "
          f"{'PASS' if steer_diff_ok else 'FAIL (need > 0.1)'}")
    if not steer_diff_ok:
        all_pass = False

    # EFE discrimination
    centered_efe = cem_results["straight_centered"]["efe"]
    offset_efes = [cem_results[k]["efe"] for k in
                   ["drifting_left", "drifting_right"]]
    mean_offset_efe = np.mean(offset_efes)
    efe_discriminates = centered_efe < mean_offset_efe
    print(f"  EFE centered={centered_efe:.3f} vs offset={mean_offset_efe:.3f} ... "
          f"{'PASS' if efe_discriminates else 'FAIL (centered should be lower)'}")
    if not efe_discriminates:
        all_pass = False

    return all_pass


def main():
    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    cfg.training.epochs = 5
    cfg.preference.warmup_epoch = 1

    set_seed(cfg.seed)
    output_dir = Path("outputs/verify_steering")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Active Inference Steering Verification")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1] Generating synthetic 4D driving data...")
    data_path = generate_data(cfg, output_dir)

    # Step 2: Train world model
    print("\n[2] Training world model (5 epochs)...")
    agent = DeepAIFAgent(cfg)
    losses = train_world_model(cfg, agent, data_path, n_epochs=5)

    # Step 3: Update preference model
    print("\n[3] Fitting preference model on successful trajectories...")
    pref_info = update_preference_model(cfg, agent, data_path)

    # Step 4: Test CEM planning
    print("\n[4] Testing CEM planning across scenarios...")
    cem_results = verify_cem_steering(cfg, agent)

    # Step 5: Check all metrics
    all_pass = check_metrics(losses, cem_results)

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see Story 4 for fixes")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

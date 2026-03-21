"""Unit tests for CARLA environment wrapper and action conversion utilities."""

import numpy as np
import pytest

from active_inference.data.carla_env import action_to_carla, carla_to_action


# --- Tests for linear throttle remap (no braking) ---
# Default: min_throttle=0.35, max_throttle=0.55
# Formula: throttle = 0.35 + 0.2 * (accel + 1) / 2

def test_action_to_carla_full_accel():
    """accel=+1.0 → max_throttle=0.55"""
    steer, throttle, brake = action_to_carla(np.array([0.5, 1.0]))
    assert steer == pytest.approx(0.5)
    assert throttle == pytest.approx(0.55)
    assert brake == pytest.approx(0.0)


def test_action_to_carla_zero_accel():
    """accel=0.0 → midpoint of [0.35, 0.55] = 0.45"""
    steer, throttle, brake = action_to_carla(np.array([0.0, 0.0]))
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.45)
    assert brake == pytest.approx(0.0)


def test_action_to_carla_full_negative():
    """accel=-1.0 → min_throttle=0.35"""
    steer, throttle, brake = action_to_carla(np.array([0.0, -1.0]))
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.35)
    assert brake == pytest.approx(0.0)


def test_action_to_carla_half_accel():
    """accel=+0.5 → 0.35 + 0.2 * 1.5/2 = 0.35 + 0.15 = 0.50"""
    steer, throttle, brake = action_to_carla(np.array([0.0, 0.5]))
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.50)
    assert brake == pytest.approx(0.0)


def test_action_to_carla_no_brake_ever():
    """No matter the accel value, brake is always 0."""
    for accel in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        _, _, brake = action_to_carla(np.array([0.0, accel]))
        assert brake == pytest.approx(0.0), f"brake != 0 for accel={accel}"


def test_action_to_carla_custom_range():
    """Test with custom min/max throttle."""
    steer, throttle, brake = action_to_carla(
        np.array([0.0, 0.0]), min_throttle=0.3, max_throttle=1.0
    )
    assert throttle == pytest.approx(0.65)  # midpoint of [0.3, 1.0]
    assert brake == pytest.approx(0.0)


# --- Clamp tests ---

def test_action_clamp():
    steer, throttle, brake = action_to_carla(np.array([2.0, 3.0]))
    assert steer == pytest.approx(1.0)
    assert throttle == pytest.approx(0.55)  # clamped to max_throttle
    assert brake == pytest.approx(0.0)

    steer, throttle, brake = action_to_carla(np.array([-5.0, -4.0]))
    assert steer == pytest.approx(-1.0)
    assert throttle == pytest.approx(0.35)  # clamped to min_throttle
    assert brake == pytest.approx(0.0)


# --- carla_to_action (inverse direction, unchanged) ---

def test_carla_to_action_throttle_only():
    result = carla_to_action(0.2, 0.6, 0.0)
    assert result == pytest.approx(np.array([0.2, 0.6]), abs=1e-6)


def test_carla_to_action_brake_only():
    result = carla_to_action(-0.5, 0.0, 0.8)
    assert result == pytest.approx(np.array([-0.5, -0.8]), abs=1e-6)


def test_carla_to_action_simultaneous():
    result = carla_to_action(0.0, 0.5, 0.3)
    assert result == pytest.approx(np.array([0.0, 0.2]), abs=1e-6)


def test_carla_to_action_clamp():
    result = carla_to_action(3.0, 2.0, 0.0)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0)

    result = carla_to_action(-3.0, 0.0, 2.0)
    assert result[0] == pytest.approx(-1.0)
    assert result[1] == pytest.approx(-1.0)


@pytest.mark.integration
def test_carla_env_reset_step():
    from active_inference.data.carla_env import CARLADrivingEnv

    env = CARLADrivingEnv()
    image, state = env.reset()
    assert image.shape == (3, 64, 64)
    assert state.shape == (4,)  # [speed, steer, heading_error, crosstrack_error]
    obs, info = env.step(np.array([0.0, 0.3]))
    assert "collision" in info
    assert "lane_invasion" in info
    env.close()

"""Unit tests for CARLA environment wrapper and action conversion utilities."""

import numpy as np
import pytest

from active_inference.data.carla_env import action_to_carla, carla_to_action


def test_action_to_carla_positive_accel():
    steer, throttle, brake = action_to_carla(np.array([0.5, 0.7]))
    assert steer == pytest.approx(0.5)
    assert throttle == pytest.approx(0.7)
    assert brake == pytest.approx(0.0)


def test_action_to_carla_negative_accel():
    steer, throttle, brake = action_to_carla(np.array([0.1, -0.3]))
    assert steer == pytest.approx(0.1)
    assert throttle == pytest.approx(0.0)
    assert brake == pytest.approx(0.3)


def test_action_to_carla_zero():
    steer, throttle, brake = action_to_carla(np.array([0.0, 0.0]))
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.0)
    assert brake == pytest.approx(0.0)


def test_carla_to_action_throttle_only():
    result = carla_to_action(0.2, 0.6, 0.0)
    assert result == pytest.approx(np.array([0.2, 0.6]), abs=1e-6)


def test_carla_to_action_brake_only():
    result = carla_to_action(-0.5, 0.0, 0.8)
    assert result == pytest.approx(np.array([-0.5, -0.8]), abs=1e-6)


def test_carla_to_action_simultaneous():
    result = carla_to_action(0.0, 0.5, 0.3)
    assert result == pytest.approx(np.array([0.0, 0.2]), abs=1e-6)


def test_action_roundtrip():
    for action_2d in [
        np.array([0.5, 0.7]),
        np.array([0.1, -0.3]),
        np.array([0.0, 0.0]),
        np.array([-0.8, 0.9]),
        np.array([0.3, -1.0]),
    ]:
        steer, throttle, brake = action_to_carla(action_2d)
        recovered = carla_to_action(steer, throttle, brake)
        assert recovered == pytest.approx(action_2d, abs=1e-6)


def test_action_clamp():
    steer, throttle, brake = action_to_carla(np.array([2.0, 3.0]))
    assert steer == pytest.approx(1.0)
    assert throttle == pytest.approx(1.0)
    assert brake == pytest.approx(0.0)

    steer, throttle, brake = action_to_carla(np.array([-5.0, -4.0]))
    assert steer == pytest.approx(-1.0)
    assert throttle == pytest.approx(0.0)
    assert brake == pytest.approx(1.0)

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
    assert state.shape == (2,)
    obs, info = env.step(np.array([0.0, 0.3]))
    assert "collision" in info
    assert "lane_invasion" in info
    env.close()

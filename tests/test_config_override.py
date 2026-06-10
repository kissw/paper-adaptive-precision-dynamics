"""Tests for Config.from_yaml CLI override mechanism."""

import pytest
from active_inference.config import Config


_DEBUG_YAML = "configs/experiment/debug.yaml"
_DEFAULT_YAML = "configs/default.yaml"


class TestFromYamlOverrides:
    def test_override_single_key(self):
        cfg = Config.from_yaml(_DEBUG_YAML,
                               overrides=["training.overshoot_horizon=5"])
        assert cfg.training.overshoot_horizon == 5

    def test_override_multiple_keys(self):
        cfg = Config.from_yaml(_DEBUG_YAML, overrides=[
            "training.overshoot_horizon=7",
            "training.overshoot_weight=1.0",
        ])
        assert cfg.training.overshoot_horizon == 7
        assert cfg.training.overshoot_weight == 1.0

    def test_no_overrides_returns_yaml_value(self):
        cfg_base = Config.from_yaml(_DEBUG_YAML)
        cfg_none = Config.from_yaml(_DEBUG_YAML, overrides=None)
        assert cfg_base.training.overshoot_horizon == cfg_none.training.overshoot_horizon

    def test_empty_overrides_list_is_noop(self):
        cfg_base = Config.from_yaml(_DEBUG_YAML)
        cfg_empty = Config.from_yaml(_DEBUG_YAML, overrides=[])
        assert cfg_base.training.lr == cfg_empty.training.lr

    def test_override_non_training_key(self):
        cfg = Config.from_yaml(_DEBUG_YAML, overrides=["seed=999"])
        assert cfg.seed == 999

    def test_default_yaml_baseline_overshoot_off(self):
        cfg = Config.from_yaml(_DEFAULT_YAML)
        assert cfg.training.overshoot_horizon == 0
        assert cfg.training.overshoot_weight == 0.0

    def test_override_enables_overshoot_on_default(self):
        cfg = Config.from_yaml(_DEFAULT_YAML, overrides=[
            "training.overshoot_horizon=5",
            "training.overshoot_weight=0.5",
        ])
        assert cfg.training.overshoot_horizon == 5
        assert cfg.training.overshoot_weight == 0.5

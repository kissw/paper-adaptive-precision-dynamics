"""Tests for checkpoint metadata added in train.py.

Verifies that:
1. build_checkpoint returns all required existing keys (world_model, optimizer, preference)
2. All metadata keys are present with correct types
3. ckpt["world_model"] is still directly loadable (backward compat)
4. config snapshot is a plain serialisable dict
5. best.pt and final.pt save correctly (smoke via tmp_path)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from active_inference.config import Config
from active_inference.agent import DeepAIFAgent

# Import helpers from train.py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("train", "scripts/train.py")
_train = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_train)

build_checkpoint      = _train.build_checkpoint
_safe_config_to_dict  = _train._safe_config_to_dict
_get_git_info         = _train._get_git_info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _debug_cfg():
    return Config.from_yaml("configs/experiment/debug.yaml")


def _fake_args(tmp_path=None):
    return SimpleNamespace(
        data="data/expert_data_v4.h5",
        output_dir=str(tmp_path or "/tmp/train_test"),
    )


def _make_ckpt(tmp_path, checkpoint_type="best", is_best=True):
    cfg = _debug_cfg()
    agent = DeepAIFAgent(cfg)
    args = _fake_args(tmp_path)
    return build_checkpoint(
        agent, cfg, args,
        epoch=3,
        global_step=300,
        train_loss=0.42,
        best_epoch=3,
        best_loss=0.42,
        checkpoint_type=checkpoint_type,
        is_best=is_best,
    )


# ---------------------------------------------------------------------------
# Test 1: Existing keys still present
# ---------------------------------------------------------------------------

def test_existing_keys_present(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    assert "world_model" in ckpt, "world_model key must be present"
    assert "optimizer"   in ckpt, "optimizer key must be present"
    assert "preference"  in ckpt, "preference key must be present"


# ---------------------------------------------------------------------------
# Test 2: Metadata keys present and have correct types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected_type", [
    ("epoch",           int),
    ("global_step",     int),
    ("best_epoch",      int),
    ("best_loss",       float),
    ("best_metric",     float),
    ("train_loss",      float),
    ("config",          dict),
    ("crop_road",       bool),
    ("image_size",      int),
    ("checkpoint_type", str),
    ("is_best",         bool),
])
def test_metadata_key_type(tmp_path, key, expected_type):
    ckpt = _make_ckpt(tmp_path)
    assert key in ckpt, f"metadata key '{key}' missing from checkpoint"
    assert isinstance(ckpt[key], expected_type), \
        f"ckpt['{key}'] should be {expected_type.__name__}, got {type(ckpt[key]).__name__}"


def test_metadata_optional_keys_present(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    for k in ("timestamp", "git_commit", "git_branch", "data_path", "output_dir", "world_model_type"):
        assert k in ckpt, f"optional metadata key '{k}' missing"


def test_checkpoint_type_values(tmp_path):
    ckpt_best  = _make_ckpt(tmp_path, checkpoint_type="best",  is_best=True)
    ckpt_final = _make_ckpt(tmp_path, checkpoint_type="final", is_best=False)
    assert ckpt_best["checkpoint_type"]  == "best"
    assert ckpt_final["checkpoint_type"] == "final"
    assert ckpt_best["is_best"]  is True
    assert ckpt_final["is_best"] is False


# ---------------------------------------------------------------------------
# Test 3: Backward compatibility — ckpt["world_model"] still loadable
# ---------------------------------------------------------------------------

def test_world_model_loadable(tmp_path):
    """ckpt['world_model'] must load correctly into a fresh world model."""
    cfg = _debug_cfg()
    # Build checkpoint from a specific agent so we know what params were saved
    agent_src = DeepAIFAgent(cfg)
    args = _fake_args(tmp_path)
    ckpt = build_checkpoint(
        agent_src, cfg, args,
        epoch=1, global_step=10, train_loss=0.5,
        best_epoch=1, best_loss=0.5,
        checkpoint_type="best", is_best=True,
    )

    # Load into a fresh agent (different random init)
    agent_dst = DeepAIFAgent(cfg)
    agent_dst.world_model.load_state_dict(ckpt["world_model"])

    # After loading, dst params must equal src params
    p_src = dict(agent_src.world_model.named_parameters())
    p_dst = dict(agent_dst.world_model.named_parameters())
    for k in p_src:
        assert torch.allclose(p_src[k], p_dst[k]), \
            f"Mismatch at {k} after loading from metadata checkpoint"


# ---------------------------------------------------------------------------
# Test 4: config is a plain serialisable dict
# ---------------------------------------------------------------------------

def test_config_is_plain_dict(tmp_path):
    cfg = _debug_cfg()
    d = _safe_config_to_dict(cfg)
    assert isinstance(d, dict), "config snapshot must be a dict"
    assert len(d) > 0, "config dict must not be empty"
    # Must be JSON-serialisable (all values are plain Python types)
    import json
    try:
        json.dumps(d)
    except (TypeError, ValueError) as e:
        pytest.fail(f"config dict is not JSON-serialisable: {e}")


def test_config_contains_encoder_fields(tmp_path):
    cfg = _debug_cfg()
    d = _safe_config_to_dict(cfg)
    assert "encoder" in d
    assert "crop_road" in d["encoder"]
    assert "image_size" in d["encoder"]


# ---------------------------------------------------------------------------
# Test 5: torch.save / torch.load round-trip preserves all keys
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    path = tmp_path / "test_ckpt.pt"
    torch.save(ckpt, str(path))

    loaded = torch.load(str(path), map_location="cpu", weights_only=False)

    for k in ("world_model", "optimizer", "preference",
              "epoch", "global_step", "best_epoch", "best_loss",
              "train_loss", "config", "crop_road", "image_size",
              "checkpoint_type", "is_best", "timestamp"):
        assert k in loaded, f"key '{k}' missing after save/load"

    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 300
    assert abs(loaded["train_loss"] - 0.42) < 1e-6
    assert loaded["checkpoint_type"] == "best"
    assert loaded["is_best"] is True


# ---------------------------------------------------------------------------
# Test 6: git info returns strings or None (never raises)
# ---------------------------------------------------------------------------

def test_git_info_does_not_raise():
    commit, branch = _get_git_info()
    assert commit is None or isinstance(commit, str)
    assert branch is None or isinstance(branch, str)


# ---------------------------------------------------------------------------
# Test 7: build_checkpoint with None optional values is safe
# ---------------------------------------------------------------------------

def test_build_checkpoint_none_values(tmp_path):
    cfg = _debug_cfg()
    agent = DeepAIFAgent(cfg)
    args = _fake_args(tmp_path)

    # best_epoch and global_step can be None early in training
    ckpt = build_checkpoint(
        agent, cfg, args,
        epoch=1,
        global_step=None,
        train_loss=None,
        best_epoch=None,
        best_loss=None,
        checkpoint_type="epoch",
        is_best=False,
    )
    assert ckpt["global_step"] is None
    assert ckpt["best_epoch"]  is None
    assert ckpt["best_loss"]   is None
    assert ckpt["train_loss"]  is None
    assert ckpt["epoch"] == 1

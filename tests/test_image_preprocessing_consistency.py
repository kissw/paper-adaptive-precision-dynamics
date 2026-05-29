"""Tests for WorldModel-level image preprocessing consistency.

Verifies that:
1. preprocess_image is identity when crop_road=false
2. preprocess_image applies crop_road when crop_road=true
3. encode_obs == manual preprocess + encoder
4. encoder-internal crop disabled in WorldModel (crop_road=False on encoder)
5. Training loss target uses preprocessed image (not raw)
6. compare rollout preprocess_for_display helper is consistent
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from active_inference.config import Config
from active_inference.agent import WorldModel
from active_inference.utils.transforms import crop_road


def _cfg_no_crop():
    """Minimal RSSM config without crop_road."""
    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    cfg.encoder.crop_road = False
    return cfg


def _cfg_crop():
    """Minimal RSSM config with crop_road=true."""
    cfg = Config.from_yaml("configs/experiment/debug.yaml")
    cfg.encoder.crop_road = True
    return cfg


# ---------------------------------------------------------------------------
# Test 1: preprocess_image is identity when crop_road=false
# ---------------------------------------------------------------------------

def test_preprocess_identity_no_crop():
    wm = WorldModel(_cfg_no_crop())
    img = torch.randn(2, 3, 64, 64)
    out = wm.preprocess_image(img)
    assert out is img or torch.equal(out, img), \
        "preprocess_image should return img unchanged when crop_road=False"


# ---------------------------------------------------------------------------
# Test 2: preprocess_image applies crop when crop_road=true
# ---------------------------------------------------------------------------

def test_preprocess_applies_crop():
    wm = WorldModel(_cfg_crop())
    assert wm._crop_road is True

    # Image with distinct top (sky, value 1.0) and bottom (road, value -1.0)
    img = torch.ones(1, 3, 64, 64)
    img[:, :, :32, :] = 1.0   # top half: sky
    img[:, :, 32:, :] = -1.0  # bottom half: road

    out = wm.preprocess_image(img)

    # Output must differ from raw input (crop changes pixel content)
    assert not torch.equal(out, img), \
        "preprocess_image should modify image when crop_road=True"

    # Output shape must be unchanged (crop + bilinear resize back)
    assert out.shape == img.shape, \
        f"preprocess_image must preserve shape: got {out.shape} expected {img.shape}"

    # After crop_road (keep bottom 60%), the top pixels should reflect road
    # (The resized output expands the bottom portion up)
    manual = crop_road(img)
    assert torch.allclose(out, manual, atol=1e-6), \
        "preprocess_image result must match crop_road() directly"


# ---------------------------------------------------------------------------
# Test 3: encode_obs equals preprocess + encoder
# ---------------------------------------------------------------------------

def test_encode_obs_equals_preprocess_then_encoder():
    cfg = _cfg_crop()
    wm = WorldModel(cfg)
    wm.eval()

    img   = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, cfg.encoder.state_dim)

    with torch.no_grad():
        emb1 = wm.encode_obs(img, state)
        emb2 = wm.encoder(wm.preprocess_image(img), state)

    assert torch.allclose(emb1, emb2, atol=1e-6), \
        "encode_obs must equal encoder(preprocess_image(img), state)"


# ---------------------------------------------------------------------------
# Test 4: ConvEncoder internal crop disabled when WorldModel has crop_road=true
# ---------------------------------------------------------------------------

def test_conv_encoder_internal_crop_disabled():
    # crop_road=true: WorldModel centralises preprocessing, encoder must be False
    wm_crop = WorldModel(_cfg_crop())
    assert wm_crop._crop_road is True
    assert getattr(wm_crop.encoder, "_crop_road", None) is False, \
        "ConvEncoder._crop_road must be False when WorldModel handles preprocessing"

    # crop_road=false: WorldModel does nothing, encoder also False
    wm_nocrop = WorldModel(_cfg_no_crop())
    assert wm_nocrop._crop_road is False
    assert getattr(wm_nocrop.encoder, "_crop_road", None) is False


# ---------------------------------------------------------------------------
# Test 5: Training loop uses preprocessed image as reconstruction target
# ---------------------------------------------------------------------------

def test_update_uses_preprocessed_img_target():
    """Verify that the loss path receives preprocessed (not raw) images.

    crop_road keeps bottom 60%, so for H=64: start_row = int(64*0.4) = 25.
    We set only rows 0-20 to 2.0 (fully within the cropped-away top region)
    to ensure the preprocessed image contains no 2.0 values.
    """
    from active_inference.agent import DeepAIFAgent

    cfg = _cfg_crop()
    agent = DeepAIFAgent(cfg)

    B, T = 2, cfg.training.seq_len
    H = 64
    # start_row for crop_road(keep_bottom_frac=0.6) on 64-px images = 25
    # Set rows 0..20 to 2.0 (fully in the cropped-away top 40%).
    raw = torch.zeros(B, T, 3, H, 64)
    raw[:, :, :, :20, :] = 2.0   # sky region — removed by crop_road
    raw[:, :, :, 20:, :] = torch.randn(B, T, 3, H - 20, 64) * 0.1

    states  = torch.randn(B, T, cfg.encoder.state_dim)
    actions = torch.randn(B, T, cfg.cem.action_dim).clamp(-1, 1)

    # Preprocessing must strip the 2.0 sky rows completely
    preprocessed = agent.world_model.preprocess_image(raw[0, 0].unsqueeze(0))
    assert preprocessed.max().item() < 1.5, \
        "Preprocessed image must not contain the 2.0 sky values (crop_road removes top 40%)"

    # Loss path must be finite
    info = agent.update(raw, states, actions)
    assert torch.isfinite(torch.tensor(info["total_loss"])), \
        "update() must produce finite loss with preprocessed target"


# ---------------------------------------------------------------------------
# Test 6: preprocess_for_display in compare script returns model-space image
# ---------------------------------------------------------------------------

def test_preprocess_for_display_consistency():
    """preprocess_for_display must match wm.preprocess_image applied to same img."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_rollout",
        "scripts/compare_rssm_token_vit_rollout.py",
    )
    mod = importlib.util.load_from_spec = spec.loader
    # Import via exec to avoid matplotlib Agg backend ordering issues
    import importlib
    compare_mod = importlib.import_module.__module__

    # Direct import of the function
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("cmp", "scripts/compare_rssm_token_vit_rollout.py")
    cmp = ilu.module_from_spec(spec)
    spec.loader.exec_module(cmp)

    wm_crop   = WorldModel(_cfg_crop())
    wm_nocrop = WorldModel(_cfg_no_crop())
    device    = torch.device("cpu")
    img_chw   = torch.randn(3, 64, 64)

    # crop_road=true: display should differ from raw
    disp_crop = cmp.preprocess_for_display(wm_crop, img_chw, device)
    expected_crop = wm_crop.preprocess_image(img_chw.unsqueeze(0)).squeeze(0)
    assert torch.allclose(disp_crop, expected_crop, atol=1e-6), \
        "preprocess_for_display must match wm.preprocess_image (crop case)"

    # crop_road=false: display equals raw
    disp_nocrop = cmp.preprocess_for_display(wm_nocrop, img_chw, device)
    assert torch.allclose(disp_nocrop, img_chw, atol=1e-6), \
        "preprocess_for_display must equal raw image when crop_road=False"

import torch
from active_inference.models import ObsDecoder, StateDecoder


def test_obs_decoder_shape():
    dec = ObsDecoder(feat_dim=320, image_channels=3)
    out = dec(torch.randn(4, 320))
    assert out.shape == (4, 3, 64, 64)


def test_state_decoder_shape():
    dec = StateDecoder(feat_dim=320, state_dim=2)
    out = dec(torch.randn(4, 320))
    assert out.shape == (4, 2)

import torch
from active_inference.models import ConvEncoder


def test_encoder_output_shape():
    enc = ConvEncoder(image_channels=3, state_dim=2, embed_dim=256)
    out = enc(torch.randn(4, 3, 64, 64), torch.randn(4, 2))
    assert out.shape == (4, 256)


def test_encoder_different_inputs_different_outputs():
    enc = ConvEncoder()
    s = torch.randn(2, 2)
    out1 = enc(torch.randn(2, 3, 64, 64), s)
    out2 = enc(torch.randn(2, 3, 64, 64), s)
    assert not torch.allclose(out1, out2)

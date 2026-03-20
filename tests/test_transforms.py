import torch
from active_inference.utils.transforms import symlog, symexp, normalize_image, denormalize_image


def test_symlog_symexp_inverse():
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)


def test_normalize_denormalize_roundtrip():
    x = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    result = denormalize_image(normalize_image(x.float()))
    assert (result.int() - x.int()).abs().max() <= 1

import torch

from active_inference.utils.bbox import (
    bbox_xyxy_raw_to_crop_road,
    bbox_xyxy_to_mask,
    bbox_xyxy_to_model_space,
    bbox_xyxy_to_token_mask,
)


def test_raw_to_crop_road_maps_y_coordinates():
    bbox = torch.tensor([10.0, 30.0, 20.0, 50.0])
    out = bbox_xyxy_raw_to_crop_road(
        bbox, image_h=64, image_w=64, keep_bottom_frac=0.6,
    )
    scale = 64.0 / 39.0
    expected = torch.tensor([10.0, (30.0 - 25.0) * scale, 20.0, (50.0 - 25.0) * scale])
    assert torch.allclose(out, expected, atol=1e-4)


def test_raw_to_crop_road_invalid_above_crop_is_nan():
    bbox = torch.tensor([10.0, 0.0, 20.0, 20.0])
    out = bbox_xyxy_raw_to_crop_road(
        bbox, image_h=64, image_w=64, keep_bottom_frac=0.6,
    )
    assert torch.isnan(out).all()


def test_model_space_mode_does_not_crop_transform():
    bbox = torch.tensor([[10.0, 30.0, 20.0, 50.0]])
    out = bbox_xyxy_to_model_space(
        bbox,
        image_h=64,
        image_w=64,
        crop_road=True,
        keep_bottom_frac=0.6,
        coord_space="model",
    )
    assert torch.equal(out, bbox)


def test_batched_time_shape_is_preserved():
    bbox = torch.tensor([[[10.0, 30.0, 20.0, 50.0], [10.0, 0.0, 20.0, 20.0]]])
    out = bbox_xyxy_raw_to_crop_road(
        bbox, image_h=64, image_w=64, keep_bottom_frac=0.6,
    )
    assert out.shape == bbox.shape
    assert torch.isfinite(out[0, 0]).all()
    assert torch.isnan(out[0, 1]).all()


def test_bbox_to_mask_invalid_returns_zero_and_false():
    bbox = torch.tensor([[10.0, 0.0, 20.0, 20.0]])
    model_bbox = bbox_xyxy_raw_to_crop_road(
        bbox, image_h=64, image_w=64, keep_bottom_frac=0.6,
    )
    mask, valid = bbox_xyxy_to_mask(model_bbox, image_h=64, image_w=64)
    assert mask.shape == (1, 1, 64, 64)
    assert mask.sum().item() == 0.0
    assert not valid.item()


def test_token_mask_expected_count_for_simple_bbox():
    bbox = torch.tensor([[0.0, 0.0, 16.0, 16.0]])
    tok = bbox_xyxy_to_token_mask(bbox, image_h=64, image_w=64, patch_size=8)
    assert tok.shape == (1, 64)
    assert tok.sum().item() == 4
    assert set(tok[0].nonzero().squeeze(-1).tolist()) == {0, 1, 8, 9}

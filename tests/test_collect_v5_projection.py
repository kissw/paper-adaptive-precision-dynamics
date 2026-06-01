"""Unit tests for camera projection utilities in collect_obstacle_data_v5.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable without installing it as a package
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_obstacle_data_v5 import (
    compute_bbox_from_corners,
    compute_visibility,
    project_world_to_image,
    _CAM_FOV,
    _CAM_FWD_OFFSET,
    _CAM_UP_OFFSET,
)

IMAGE_SIZE = 64
CX = CY = IMAGE_SIZE / 2.0          # 32.0
F = CX / math.tan(math.radians(_CAM_FOV) / 2.0)  # 32.0 for 90° FOV


# ─────────────────────────────────────────────────────────────────────────────
# project_world_to_image
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectWorldToImage:
    """Verify perspective projection math against analytical values."""

    def test_object_directly_in_front_center_of_image(self):
        """Object directly ahead at camera height projects to image centre."""
        # Ego at origin, yaw=0 (facing +X), object 10 m ahead at camera height
        u, v = project_world_to_image(
            point_world_xyz=(10.0, 0.0, _CAM_UP_OFFSET),
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert u is not None and v is not None
        assert abs(u - CX) < 1e-6, f"Expected u={CX}, got {u}"
        assert abs(v - CY) < 1e-6, f"Expected v={CY}, got {v}"

    def test_object_right_of_centre(self):
        """Object 1 m to the right of the camera axis appears right of centre."""
        # right direction: (sin 0, -cos 0) = (0, -1) → world y=-1 is right
        u, v = project_world_to_image(
            point_world_xyz=(10.0, -1.0, _CAM_UP_OFFSET),
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert u is not None
        Z_cam = 10.0 - _CAM_FWD_OFFSET          # 8.5
        X_cam = 0.0 * 0.0 - (-1.0) * 1.0       # right_body = sin(0)*dx - cos(0)*dy = 1
        expected_u = CX + F * X_cam / Z_cam
        assert abs(u - expected_u) < 1e-5, f"Expected u≈{expected_u:.4f}, got {u:.4f}"
        assert u > CX, "Right object should appear right of centre (u > cx)"

    def test_object_above_camera_height_appears_above_centre(self):
        """Object above camera height (smaller v) appears in upper half."""
        u, v = project_world_to_image(
            point_world_xyz=(10.0, 0.0, _CAM_UP_OFFSET + 1.0),
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert v is not None
        assert v < CY, "Object above camera height should have v < cy (upper image)"

    def test_object_below_camera_height_appears_below_centre(self):
        """Ground-level object below camera height appears in lower half."""
        u, v = project_world_to_image(
            point_world_xyz=(10.0, 0.0, 0.0),  # ground level
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert v is not None
        assert v > CY, "Ground-level object should appear in lower half (v > cy)"

    def test_behind_camera_returns_none(self):
        """Point behind the camera (Z_cam <= 0) returns (None, None)."""
        u, v = project_world_to_image(
            point_world_xyz=(-5.0, 0.0, _CAM_UP_OFFSET),  # 5 m behind ego
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert u is None and v is None

    def test_at_camera_position_returns_none(self):
        """Point at the camera focal plane (Z_cam = 0) returns (None, None)."""
        u, v = project_world_to_image(
            point_world_xyz=(_CAM_FWD_OFFSET, 0.0, _CAM_UP_OFFSET),
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=0.0,
            image_size=IMAGE_SIZE,
        )
        assert u is None and v is None

    def test_analytical_u_formula(self):
        """u = f * right_body / (fwd_body - cam_fwd) + cx."""
        obj = (20.0, -2.0, _CAM_UP_OFFSET)
        ego = (0.0, 0.0, 0.0)
        yaw = 0.0
        u, v = project_world_to_image(obj, ego, yaw, IMAGE_SIZE)

        fwd_body = 20.0 * math.cos(0) + (-2.0) * math.sin(0)       # 20
        right_body = 20.0 * math.sin(0) - (-2.0) * math.cos(0)     # 2
        Z_cam = fwd_body - _CAM_FWD_OFFSET                          # 18.5
        X_cam = right_body                                           # 2
        expected_u = F * X_cam / Z_cam + CX
        assert abs(u - expected_u) < 1e-5

    def test_non_zero_yaw_projects_correctly(self):
        """Object on the forward axis for a rotated vehicle projects to centre."""
        # Ego facing +Y (yaw=90°). Object at (0, 10, cam_up).
        yaw = 90.0
        yaw_rad = math.radians(yaw)
        fwd_body = 0.0 * math.cos(yaw_rad) + 10.0 * math.sin(yaw_rad)   # 10
        right_body = 0.0 * math.sin(yaw_rad) - 10.0 * math.cos(yaw_rad) # 0
        # Object is 10 m directly ahead of rotated vehicle → should project to centre
        u, v = project_world_to_image(
            point_world_xyz=(0.0, 10.0, _CAM_UP_OFFSET),
            ego_pos_xyz=(0.0, 0.0, 0.0),
            ego_yaw_deg=yaw,
            image_size=IMAGE_SIZE,
        )
        assert u is not None and v is not None
        assert abs(u - CX) < 1e-5, f"Expected u centre, got {u}"
        assert abs(v - CY) < 1e-5, f"Expected v centre, got {v}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_bbox_from_corners
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBboxFromCorners:

    def test_all_corners_behind_camera_returns_none(self):
        us = [None, None, None, None, None, None, None, None]
        vs = [None, None, None, None, None, None, None, None]
        assert compute_bbox_from_corners(us, vs, IMAGE_SIZE) is None

    def test_all_corners_outside_image_returns_none(self):
        # All corners to the right of the image (u > W)
        us = [100.0] * 8
        vs = [32.0] * 8
        assert compute_bbox_from_corners(us, vs, IMAGE_SIZE) is None

    def test_valid_corners_return_clipped_bbox(self):
        us = [10.0, 50.0, 10.0, 50.0, 10.0, 50.0, 10.0, 50.0]
        vs = [15.0, 45.0, 15.0, 45.0, 15.0, 45.0, 15.0, 45.0]
        bbox = compute_bbox_from_corners(us, vs, IMAGE_SIZE)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == pytest.approx(10.0)
        assert y1 == pytest.approx(15.0)
        assert x2 == pytest.approx(50.0)
        assert y2 == pytest.approx(45.0)

    def test_corners_partially_outside_are_clipped(self):
        us = [-5.0, 40.0]
        vs = [20.0, 70.0]  # y2 > H → clipped to 64
        bbox = compute_bbox_from_corners(us, vs, IMAGE_SIZE)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 == pytest.approx(0.0)
        assert y2 == pytest.approx(float(IMAGE_SIZE))

    def test_mixed_valid_and_none_corners(self):
        us = [None, 30.0, None, 50.0]
        vs = [None, 20.0, None, 40.0]
        bbox = compute_bbox_from_corners(us, vs, IMAGE_SIZE)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert abs(x1 - 30.0) < 1e-6
        assert abs(x2 - 50.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# compute_visibility
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeVisibility:

    def _good_bbox(self):
        """A bbox that passes all geometric checks: area=760, y2=54 > 0.4*64=25.6."""
        return (5.0, 35.0, 45.0, 54.0)   # area = 40 * 19 = 760

    def test_visible_when_all_conditions_met(self):
        assert compute_visibility(
            in_front=True,
            distance=20.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
        )

    def test_not_visible_when_behind(self):
        assert not compute_visibility(
            in_front=False,
            distance=20.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
        )

    def test_not_visible_when_distance_exceeds_threshold(self):
        assert not compute_visibility(
            in_front=True,
            distance=51.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
            dist_threshold=50.0,
        )

    def test_not_visible_at_exact_distance_threshold(self):
        assert not compute_visibility(
            in_front=True,
            distance=50.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
            dist_threshold=50.0,
        )

    def test_not_visible_when_bbox_is_none(self):
        assert not compute_visibility(
            in_front=True,
            distance=10.0,
            bbox=None,
            image_size=IMAGE_SIZE,
        )

    def test_not_visible_when_bbox_area_too_small(self):
        # Area = 4 * 4 = 16 < 20
        small_bbox = (30.0, 30.0, 34.0, 34.0)
        assert not compute_visibility(
            in_front=True,
            distance=10.0,
            bbox=small_bbox,
            image_size=IMAGE_SIZE,
            bbox_area_threshold=20.0,
        )

    def test_not_visible_when_y2_at_or_below_crop_boundary(self):
        """y2 <= 0.4 * image_size → in cropped-away region → not visible."""
        # 0.4 * 64 = 25.6; y2 = 24 < 25.6
        crop_bbox = (5.0, 5.0, 45.0, 24.0)  # area = 40*19 = 760 >> 20
        assert not compute_visibility(
            in_front=True,
            distance=10.0,
            bbox=crop_bbox,
            image_size=IMAGE_SIZE,
        )

    def test_not_visible_y2_exactly_at_crop_boundary(self):
        """y2 = 0.4 * image_size is not strictly greater → not visible."""
        crop_bbox = (5.0, 5.0, 45.0, 25.6)
        assert not compute_visibility(
            in_front=True,
            distance=10.0,
            bbox=crop_bbox,
            image_size=IMAGE_SIZE,
        )

    def test_visible_y2_just_above_crop_boundary(self):
        """y2 = 0.4 * image_size + ε → visible (other conditions met)."""
        bbox = (5.0, 5.0, 45.0, 26.0)   # y2=26 > 25.6
        assert compute_visibility(
            in_front=True,
            distance=10.0,
            bbox=bbox,
            image_size=IMAGE_SIZE,
        )

    def test_custom_thresholds_are_respected(self):
        assert not compute_visibility(
            in_front=True,
            distance=30.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
            dist_threshold=25.0,  # stricter
        )
        assert compute_visibility(
            in_front=True,
            distance=30.0,
            bbox=self._good_bbox(),
            image_size=IMAGE_SIZE,
            dist_threshold=35.0,  # looser
        )

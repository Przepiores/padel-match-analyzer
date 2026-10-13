import numpy as np
import pytest

from padel.court.geometry import Court

# A synthetic camera: the court seen with perspective (far side narrower).
CORNERS_PX = [(100.0, 700.0), (900.0, 700.0), (650.0, 200.0), (350.0, 200.0)]


def test_from_corners_maps_corners_to_metres():
    court = Court.from_corners(CORNERS_PX)
    assert court.to_court(*CORNERS_PX[0]) == pytest.approx((0, 0), abs=1e-6)
    assert court.to_court(*CORNERS_PX[2]) == pytest.approx((10, 20), abs=1e-6)
    assert court.to_image(10, 0) == pytest.approx(CORNERS_PX[1], abs=1e-6)


def test_from_points_without_near_corners():
    full = Court.from_corners(CORNERS_PX)
    landmarks_m = [(5.0, 3.05), (0.0, 10.0), (10.0, 10.0), (0.0, 20.0), (10.0, 20.0)]
    court = Court.from_points([full.to_image(*p) for p in landmarks_m], landmarks_m)
    assert court.to_court(*CORNERS_PX[0]) == pytest.approx((0, 0), abs=1e-6)


def test_from_points_rejects_too_few():
    with pytest.raises(ValueError):
        Court.from_points([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])


def test_contains_and_sides():
    court = Court.from_corners(CORNERS_PX)
    near, far = court.to_image(5, 2), court.to_image(5, 18)
    assert court.contains(*near) and court.contains(*far)
    assert not court.contains(*court.to_image(5, -3))
    assert court.contains(*court.to_image(5, -1), margin_m=1.5)
    assert court.is_near_side(*near) and not court.is_near_side(*far)


def test_ball_region_excludes_image_edges():
    court = Court.from_corners(CORNERS_PX)
    mask = court.ball_region(1000, 800, margin_px=10)
    assert mask[100, 500] == 1  # above the court centre
    assert mask[100, 5] == 0  # left image edge, beside the court
    assert mask.dtype == np.uint8

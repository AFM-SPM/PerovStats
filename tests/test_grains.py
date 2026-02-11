import pytest
import numpy as np
from skimage.measure import regionprops, label

from perovstats.grains import (
    tidy_border,
    find_grains,
    find_median_grain_size,
    find_mean_grain_size,
    find_mode_grain_size,
    find_circularity_rating,
)

def test_find_grains(dummy_perovstats_object):
    find_grains(dummy_perovstats_object.config, dummy_perovstats_object.images[0], 0)

    assert dummy_perovstats_object.images[0].grains is not None
    assert len(dummy_perovstats_object.images[0].grains) == 4


values = [0, 1, 1, 2, 3, 4, 5]
expected_median = 2
expected_mean = 2.2857142857142856
expected_mode = 1

def test_find_median_grain_size():
    median = find_median_grain_size(values)
    assert median == expected_median


def test_find_mean_grain_size():
    mean = find_mean_grain_size(values)
    assert mean == expected_mean


def test_find_mode_grain_size():
    mode = find_mode_grain_size(values)
    assert mode == expected_mode


def test_tidy_borders(dummy_mask: np.ndarray):
    mask = dummy_mask
    new_mask = tidy_border(mask)

    assert new_mask.shape == mask.shape
    assert new_mask.dtype == mask.dtype


@pytest.mark.parametrize(
        ("grain_mask", "expected_rating"),
        [
            pytest.param(
                np.array([
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                ]),
                0.9595052218168317,
                id="circle"
            ),
            pytest.param(
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                ]),
                0.43116456976745576,
                id="non-circle"
            ),
        ]
)
def test_find_circularity_rating(grain_mask, expected_rating):
    labeled = label(grain_mask)
    props = regionprops(labeled)[0]
    area = props.area
    perimeter = props.perimeter_crofton

    circularity_rating = find_circularity_rating(area, perimeter)

    assert circularity_rating == expected_rating

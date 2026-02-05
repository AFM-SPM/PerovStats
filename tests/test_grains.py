import pytest
import numpy as np

from perovstats.grains import (
    tidy_border,
    find_grains,
    find_median_grain_size,
    find_mean_grain_size,
    find_mode_grain_size
)

def test_find_grains(dummy_perovstats_object):
    find_grains(dummy_perovstats_object)

    assert dummy_perovstats_object.images[0].grains is not None


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


def test_tidy_borders(mask_random):
    mask = mask_random
    new_mask = tidy_border(mask)

    assert new_mask.shape == mask.shape
    assert new_mask.dtype == mask.dtype

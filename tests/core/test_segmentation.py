import numpy as np

from perovstats.core.segmentation import (
    segment_image,
    clean_mask,
    create_grain_mask,
    create_frequency_mask,
    tidy_border
)
from perovstats.core.classes import PerovStats


def test_segment_image(dummy_perovstats_object: PerovStats):
    image_data = dummy_perovstats_object.images[0]

    segment_image(dummy_perovstats_object.config, image_data)

    assert image_data.mask.shape == image_data.image_original.shape


def test_create_grain_mask(dummy_original_image: np.ndarray) -> None:
    """Test creating a grain mask."""
    x = create_grain_mask(
        dummy_original_image,
        threshold_block_size=55,
        threshold_offset=0,
        smooth_sigma=8,
        area_threshold=10000,
        disk_radius=40,
        height_bias=100
    )
    assert x.shape == dummy_original_image.shape
    assert x.dtype == np.dtype(bool)


def test_clean_mask(dummy_mask: np.ndarray) -> None:
    """Test cleaning a grain mask."""
    x = clean_mask(dummy_mask)
    assert x.shape == dummy_mask.shape
    assert x.dtype == np.dtype(bool)


def test_create_frequency_mask(dummy_original_image) -> None:
    shape = dummy_original_image.shape

    freq_grid = create_frequency_mask(dummy_original_image)

    assert freq_grid.shape == shape
    assert freq_grid[0,0] == 0.0
    assert np.isclose(freq_grid[0,1], freq_grid[0,-1])
    assert np.isclose(freq_grid[1,0], freq_grid[-1,0])
    half_shape = (round(shape[0]/2), round(shape[1]/2))
    assert np.isclose(freq_grid[0,half_shape[1]], 1.0)
    assert np.isclose(freq_grid[half_shape[0],0], 1.0)
    assert np.isclose(freq_grid[half_shape[0],half_shape[1]], np.sqrt(2))


def test_tidy_borders(dummy_mask: np.ndarray):
    mask = dummy_mask
    new_mask = tidy_border(mask)

    assert new_mask.shape == mask.shape
    assert new_mask.dtype == mask.dtype

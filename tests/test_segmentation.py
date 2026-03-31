import numpy as np

from perovstats.segmentation import (
    segment_image,
    clean_mask,
    create_grain_mask
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

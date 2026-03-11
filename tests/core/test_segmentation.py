import numpy as np
import pytest

from perovstats.core.segmentation import (
    clean_mask,
    create_grain_mask,
    create_frequency_mask,
    tidy_border
)


def test_create_grain_mask(dummy_original_image: np.ndarray) -> None:
    """Test creating a grain mask."""
    x = create_grain_mask(dummy_original_image, threshold_block_size=55, smooth_sigma=8, area_threshold=10000, disk_radius=40)
    assert x.shape == dummy_original_image.shape
    assert x.dtype == np.dtype(bool)


def test_clean_mask(dummy_mask: np.ndarray) -> None:
    """Test cleaning a grain mask."""
    x = clean_mask(dummy_mask)
    assert x.shape == dummy_mask.shape
    assert x.dtype == np.dtype(bool)


# @pytest.mark.parametrize(
#         (
#             "filename",
#             "threshold_block_size",
#             "smooth_sigma",
#             "area_threshold",
#             "disk_radius",
#             "pixel_to_nm_scaling",
#             "expected"
#         ),
#         [
#             pytest.param(
#                 "dummy_filename",
#                 55,
#                 8,
#                 10000,
#                 40,
#                 19.53125,
#                 1.92,
#                 id="mad thresholding"
#             )
#         ]
# )
# def test_find_threshold(
#     filename: str,
#     dummy_high_pass,
#     threshold_block_size: int,
#     smooth_sigma: float,
#     area_threshold: float,
#     disk_radius: float,
#     pixel_to_nm_scaling: float,
#     expected: float,
# ):
#     area_threshold = area_threshold / (pixel_to_nm_scaling**2)
#     disk_radius = disk_radius / pixel_to_nm_scaling

#     threshold = find_threshold(
#         filename,
#         dummy_high_pass,
#         threshold_block_size=threshold_block_size
#         pixel_to_nm_scaling,
#         smooth_sigma,
#         area_threshold,
#         disk_radius,
#     )

#     assert threshold == expected


@pytest.mark.parametrize(
    ("shape", "cutoff", "width"),
    [
        (
            (512, 512),
            0.5,
            0,
        ),
        (
            (256, 512),
            0.5,
            0,
        ),
        (
            (512, 512),
            0.5,
            0.1,
        ),
    ],
)
def test_create_frequency_mask(shape: tuple, cutoff: float, width: float) -> None:
    """Test creating a frequency mask."""
    x = create_frequency_mask(shape, cutoff=cutoff, edge_width=width)

    assert x.shape == shape


def test_tidy_borders(dummy_mask: np.ndarray):
    mask = dummy_mask
    new_mask = tidy_border(mask)

    assert new_mask.shape == mask.shape
    assert new_mask.dtype == mask.dtype

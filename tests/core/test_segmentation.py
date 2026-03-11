import numpy as np
import pytest
import skimage as ski

from perovstats.core.segmentation import (
    clean_mask,
    create_grain_mask,
    threshold_mad,
    threshold_mean_std,
    find_threshold,
    create_frequency_mask,
    tidy_border
)

@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_threshold_mean_std(dummy_original_image: np.ndarray, k: float) -> None:
    """Test mean/std threshold."""
    x = threshold_mean_std(dummy_original_image, k)
    assert x == dummy_original_image.mean() + k * dummy_original_image.std()


@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_threshold_mad(dummy_original_image: np.ndarray, k: float) -> None:
    """Test median + mad threshold."""
    x = threshold_mad(dummy_original_image, k=k)
    med = np.median(dummy_original_image)
    mad = np.median(np.abs(dummy_original_image.astype(np.float32) - med))
    assert x == med + mad * k * 1.4826


def test_create_grain_mask(dummy_original_image: np.ndarray) -> None:
    """Test creating a grain mask."""
    x = create_grain_mask(dummy_original_image, threshold=3)
    assert x.shape == dummy_original_image.shape
    assert x.dtype == np.dtype(bool)


def test_clean_mask(dummy_mask: np.ndarray) -> None:
    """Test cleaning a grain mask."""
    x = clean_mask(dummy_mask)
    assert x.shape == dummy_mask.shape
    assert x.dtype == np.dtype(bool)


@pytest.mark.parametrize(
        (
            "filename",
            "threshold_func",
            "smooth_sigma",
            "smooth_func",
            "area_threshold",
            "disk_radius",
            "min_threshold",
            "max_threshold",
            "pixel_to_nm_scaling",
            "expected"
        ),
        [
            pytest.param(
                "dummy_filename",
                threshold_mad,
                8,
                ski.filters.gaussian,
                10000,
                40,
                0,
                4,
                19.53125,
                1.92,
                id="mad thresholding"
            ),
            pytest.param(
                "dummy_filename",
                threshold_mean_std,
                8,
                ski.filters.gaussian,
                10000,
                40,
                0,
                4,
                19.53125,
                2.0,
                id="std thresholding"
            )
        ]
)
def test_find_threshold(
    filename: str,
    dummy_high_pass,
    threshold_func: callable,
    smooth_sigma: float,
    smooth_func: callable,
    area_threshold: float,
    disk_radius: float,
    min_threshold: float,
    max_threshold: float,
    pixel_to_nm_scaling: float,
    expected: float,
):
    area_threshold = area_threshold / (pixel_to_nm_scaling**2)
    disk_radius = disk_radius / pixel_to_nm_scaling

    threshold = find_threshold(
        filename,
        dummy_high_pass,
        pixel_to_nm_scaling,
        threshold_func,
        smooth_sigma,
        smooth_func,
        area_threshold,
        disk_radius,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

    assert threshold == expected


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

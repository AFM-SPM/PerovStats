import numpy as np
import pytest

from perovstats.segmentation import clean_mask, create_grain_mask, threshold_mad, threshold_mean_std

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

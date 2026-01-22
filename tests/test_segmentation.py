import numpy as np
import pytest
from perovstats.segmentation import clean_mask, create_grain_mask, threshold_mad, threshold_mean_std

@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_threshold_mean_std(image_random: np.ndarray, k: float) -> None:
    """Test mean/std threshold."""
    x = threshold_mean_std(image_random, k)
    assert x == image_random.mean() + k * image_random.std()


@pytest.mark.parametrize(
    "k",
    [1, 2, 3],
)
def test_threshold_mad(image_random: np.ndarray, k: float) -> None:
    """Test median+mad threshold."""
    x = threshold_mad(image_random, k=k)
    med = np.median(image_random)
    mad = np.median(np.abs(image_random.astype(np.float32) - med))
    assert x == med + mad * k * 1.4826


def test_create_grain_mask(image_random: np.ndarray) -> None:
    """Test creating a grain mask."""
    x = create_grain_mask(image_random, threshold=3)
    assert x.shape == image_random.shape
    assert x.dtype == np.dtype(bool)


def test_clean_mask(mask_random: np.ndarray) -> None:
    """Test cleaning a grain mask."""
    x = clean_mask(mask_random)
    assert x.shape == mask_random.shape
    assert x.dtype == np.dtype(bool)

import numpy as np
import pytest

from perovstats.core.classes import PerovStats, ImageData
from perovstats.fourier import (
    perform_fourier,
    find_cutoff,
    split_frequencies,
    apply_cutoff,
    create_frequency_mask
)


@pytest.mark.parametrize(
    ("image"),
    [
        pytest.param(
            np.array([
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1]
            ]),
            id="Successful frequency split"
        ),
    ]
)
def test_split_frequencies(image, dummy_perovstats_object: PerovStats):
    config = dummy_perovstats_object.config
    image_data = dummy_perovstats_object.images[0]
    image_data.high_pass = None
    image_data.low_pass = None
    config["fourier"]["cutoff_bounds"] = [0, 10]
    config["fourier"]["cutoff_step"] = 1
    config["fourier"]["min_rms"] = 0

    split_frequencies(config, image_data)

    high_pass = image_data.high_pass
    low_pass = image_data.low_pass
    image = image_data.image_original

    assert high_pass.shape == image.shape
    assert low_pass.shape == image.shape
    assert np.allclose((high_pass + low_pass), image)


def test_perform_fourier(dummy_original_image: np.ndarray):
    """Test splitting an image between background and foreground patterns."""
    high_pass, low_pass = perform_fourier(dummy_original_image, cutoff=1.0, edge_width=0.03)

    assert high_pass.shape == dummy_original_image.shape
    assert low_pass.shape == dummy_original_image.shape
    assert np.allclose((high_pass + low_pass), dummy_original_image)


@pytest.mark.parametrize(
    ("edge_width", "min_cutoff", "max_cutoff", "min_rms", "pixel_to_nm_scaling", "expected"),
    [
        pytest.param(
            0.03,
            0,
            10,
            0,
            1,
            1.2982177734375,
            id="Successful cutoff calculation"
        ),
        pytest.param(
            0.03,
            0,
            10,
            1000,
            1,
            None,
            id="No cutoff found"
        ),
    ]
)
def test_find_cutoff(
    dummy_image_data_object: ImageData,
    edge_width: float,
    min_cutoff: float,
    max_cutoff: float,
    min_rms: float,
    pixel_to_nm_scaling: float,
    expected: float | None,
):
    """Test finding in ideal cutoff for given image."""
    cutoff = find_cutoff(
        image_object=dummy_image_data_object,
        edge_width=edge_width,
        min_cutoff=min_cutoff,
        max_cutoff=max_cutoff,
        min_rms=min_rms,
        pixel_to_nm_scaling=pixel_to_nm_scaling
    )
    assert cutoff == expected


def test_apply_cutoff_hard_threshold() -> None:
    """Test creating a frequency mask."""
    f_grid = np.linspace(0, 1, 100).reshape(10, 10)
    cutoff = 0.5
    edge_width = 0
    mask = apply_cutoff(f_grid, cutoff=cutoff, edge_width=edge_width)

    assert mask.shape == (10, 10)
    assert np.all(np.isin(mask, [0,1]))
    assert mask[f_grid < cutoff].max() == 0
    assert mask[f_grid >= cutoff].min() == 1


def test_apply_cutoff_soft_threshold() -> None:
    """Test creating a frequency mask."""
    f_grid = np.linspace(0, 1, 100).reshape(10, 10)
    cutoff = 0.5
    edge_width = 0.1
    mask = apply_cutoff(f_grid, cutoff=cutoff, edge_width=edge_width)

    assert mask.shape == (10, 10)
    assert mask.min() >= 0
    assert mask.max() <= 1
    assert np.any((mask > 0) & (mask < 1))
    # Find midpoint and ensure the value is ~0.5
    idx = np.unravel_index(np.argmin(np.abs(f_grid - cutoff)), f_grid.shape)
    assert np.isclose(mask[idx], 0.5, atol=0.1)


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

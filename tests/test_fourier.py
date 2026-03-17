import numpy as np
import pytest

from perovstats.core.classes import PerovStats, ImageData
from perovstats.fourier import (
    perform_fourier,
    find_cutoff,
    split_frequencies,
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
    config["freqsplit"]["cutoff_bounds"] = [0, 10]
    config["freqsplit"]["cutoff_step"] = 1
    config["freqsplit"]["min_rms"] = 0

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
    ("edge_width", "min_cutoff", "max_cutoff", "cutoff_step", "min_rms", "pixel_to_nm_scaling", "expected"),
    [
        pytest.param(
            0.03,
            0,
            10,
            1,
            0,
            1,
            1,
            id="Successful cutoff calculation"
        ),
        pytest.param(
            0.03,
            0,
            10,
            1,
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
    cutoff_step: float,
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
        cutoff_step=cutoff_step,
        min_rms=min_rms,
        pixel_to_nm_scaling=pixel_to_nm_scaling
    )
    assert cutoff == expected

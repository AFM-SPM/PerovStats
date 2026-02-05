import numpy as np
import pytest
import skimage as ski

from perovstats.classes import PerovStats
from perovstats.fourier import split_frequencies, create_masks, normalise_array, find_threshold
from perovstats.segmentation import threshold_mad, threshold_mean_std

def test_create_masks(dummy_perovstats_object: PerovStats, image_random):
    image_data = dummy_perovstats_object.images[0]
    image_data.image_original = image_random
    image_data.high_pass = None
    image_data.low_pass = None

    create_masks(dummy_perovstats_object.config, image_data)

    assert image_data.mask.shape == image_data.image_original.shape


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


@pytest.mark.parametrize(
        ("arr", "expected"),
        [
            pytest.param(
                np.array([
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]
                ]),
                np.array([
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]
                ]),
                id="bool array, no change"
            ),
            pytest.param(
                np.array([
                    [0.1, 0.6, 0.8],
                    [0.8, 0, 0.4],
                    [0.2, 0.9, 0.3]
                ]),
                np.array([
                    [0.11076512, 0.66681495, 0.88923488],
                    [0.88923488, 0, 0.44439502],
                    [0.22197509, 1, 0.33318505]
                ]),
                id="float array"
            )
        ]
)
def test_normalise_array(arr: np.array, expected: np.array):
    norm_arr = normalise_array(arr)

    assert np.allclose(norm_arr, expected)


@pytest.mark.parametrize(
        (
            "filename",
            "threshold_func",
            "smooth_sigma",
            "smooth_func",
            "area_threshold",
            "disk_radius",
            "pixel_to_nm_scaling",
            "min_threshold",
            "max_threshold",
            "threshold_step",
            "expected"
        ),
        [
            pytest.param(
                "dummy_filename",
                threshold_mad,
                8,
                ski.filters.gaussian,
                1000,
                4,
                0.01,
                0,
                1,
                0.5,
                0.5,
                id="mad thresholding function"
            ),
            pytest.param(
                "dummy_filename",
                threshold_mad,
                8,
                ski.filters.gaussian,
                10000,
                40,
                1,
                0,
                5,
                0.5,
                4.5,
                id="std thresholding function"
            )
        ]
)
def test_find_threshold(
    filename: str,
    basic_grained_image: np.ndarray,
    threshold_func: callable,
    smooth_sigma: float,
    smooth_func: callable,
    area_threshold: float,
    disk_radius: float,
    pixel_to_nm_scaling: float,
    min_threshold: float,
    max_threshold: float,
    threshold_step: float,
    expected: float,
):
    threshold = find_threshold(
        filename,
        basic_grained_image,
        threshold_func,
        smooth_sigma,
        smooth_func,
        area_threshold,
        disk_radius,
        pixel_to_nm_scaling,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        threshold_step=threshold_step,
    )

    assert threshold == expected

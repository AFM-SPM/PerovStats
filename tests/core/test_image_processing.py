import pytest
import numpy as np

from perovstats.core.image_processing import (
    calculate_rms,
    normalise_array,
    extend_image,
    get_horizontal_gradients
)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(
            np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]),
            0,
            id="No roughness"
        ),
        pytest.param(
            np.array([
                [0, 0.1, 0.6],
                [0.2, 0.5, 0.4],
                [0.1, 0.3, 0.2]
            ]),
            0.32659863237109044,
            id="Medium roughness"
        )
    ]
)
def test_calculate_rms(image: np.array, expected: float):
    rms = calculate_rms(image)
    assert rms == expected


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


def test_extend_image(dummy_original_image: np.ndarray) -> None:
    """Test extending an image."""
    extended_image, extent = extend_image(image=dummy_original_image)

    rows, cols = dummy_original_image.shape
    assert isinstance(extent, dict)
    assert extent["top"] == rows // 2
    assert extent["bottom"] == rows // 2
    assert extent["left"] == cols // 2
    assert extent["right"] == cols // 2

    assert extended_image.shape == (
        rows + extent["top"] + extent["bottom"],
        cols + extent["left"] + extent["right"],
    )


def test_extend_image_not_implemented_error(dummy_original_image: np.ndarray) -> None:
    """Test NotImplementedError is raised if method != cv2.BORDER_REFLECT."""
    with pytest.raises(NotImplementedError):
        extend_image(image=dummy_original_image, method=0)


def test_get_horizontal_gradients(dummy_low_pass, dummy_low_pass_h_gradient):
    threshold = 50
    mask = get_horizontal_gradients(dummy_low_pass, threshold)

    assert np.array_equal(mask, dummy_low_pass_h_gradient)

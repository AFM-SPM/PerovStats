import pytest
import numpy as np

from perovstats.classes import ImageData
from perovstats.freqsplit import (
    extend_image,
    create_frequency_mask,
    frequency_split,
    find_cutoff,
    calculate_rms
)

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


def test_frequency_split(dummy_original_image):
    """Test splitting an image between background and foreground patterns."""
    high_pass, low_pass = frequency_split(dummy_original_image, cutoff=1, edge_width=0.03)

    assert high_pass.shape == dummy_original_image.shape
    assert low_pass.shape == dummy_original_image.shape
    assert np.allclose((high_pass + low_pass), dummy_original_image)


@pytest.mark.parametrize(
    ("edge_width", "min_cutoff", "max_cutoff", "cutoff_step", "min_rms", "expected"),
    [
        pytest.param(
            0.03,
            0,
            10,
            1,
            0,
            1,
            id="Successful cutoff calculation"
        ),
        pytest.param(
            0.03,
            0,
            10,
            1,
            1000,
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
    )
    assert cutoff == expected


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

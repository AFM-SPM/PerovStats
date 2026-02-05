import matplotlib
matplotlib.use("Agg")

from yaml import safe_load
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pytest

from perovstats.classes import Grain, ImageData, PerovStats

BASE_DIR = Path.cwd()

@pytest.fixture
def basic_grained_image() -> np.array:
    """Basic image containing a findable grain

    Returns
    -------
    np.array
        Basic image with a grain that can be found
    """
    return np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ])

@pytest.fixture
def image_random() -> npt.NDArray:
    """Random image as NumPy array.

    Returns
    -------
    npt.NDArray
        Random 2 dimensional array

    """
    rng = np.random.default_rng(seed=1000)
    return rng.random((16, 16))


@pytest.fixture
def mask_random(image_random: np.ndarray) -> npt.NDArray:
    """Random mask as NumPy array.

    Returns
    -------
    npt.NDArray
        Random 2 dimensional boolean array

    """
    return image_random > 0.5


@pytest.fixture
def default_config() -> dict:
    config_path = BASE_DIR / "src" / "perovstats" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = safe_load(f)
    config["freqsplit"]["cutoff"] = 1
    return config


@pytest.fixture
def dummy_grain() -> Grain:
    grain = Grain(
        grain_id=0,
        grain_area=10.2,
        grain_circularity_rating=0.6
    )
    return grain


@pytest.fixture
def dummy_image_data(mask_random, image_random, dummy_grain, tmp_path) -> ImageData:
    image_data = ImageData(
        image_original=image_random,
        mask=mask_random,
        high_pass=image_random,
        low_pass=image_random,
        grains={0: dummy_grain},
        file_directory="file/dir",
        filename="dummy_filename",
        mask_rgb=mask_random,
        grains_per_nm2=2,
        mask_size_x_nm=10,
        mask_size_y_nm=10,
        mean_grain_size=None,
        median_grain_size=None,
        mode_grain_size=None,
        mask_area_nm=100,
        num_grains=1,
        cutoff_freq_nm=1.32,
        cutoff=0.9,
        pixel_to_nm_scaling=1,
        threshold=-0.8,
    )
    return image_data


@pytest.fixture
def dummy_perovstats_object(dummy_image_data, default_config) -> PerovStats:
    perovstats_object = PerovStats(
        images=[dummy_image_data],
        config=default_config
    )
    return perovstats_object

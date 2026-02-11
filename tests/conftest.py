import matplotlib
matplotlib.use("Agg")

from yaml import safe_load
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pytest
import pickle

from perovstats.classes import Grain, ImageData, PerovStats

BASE_DIR = Path.cwd()


@pytest.fixture
def default_config() -> dict:
    config_path = BASE_DIR / "src" / "perovstats" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = safe_load(f)
    config["freqsplit"]["cutoff"] = 1
    return config


@pytest.fixture
def dummy_grain_mask() -> np.ndarray:
    with open('./tests/resources/single_grain_mask.pkl', 'rb') as file:
        arr = pickle.load(file)
    return arr


@pytest.fixture
def dummy_original_image() -> np.ndarray:
    with open('./tests/resources/small_original.pkl', 'rb') as file:
        arr = pickle.load(file)
    return arr


@pytest.fixture
def dummy_high_pass() -> np.ndarray:
    with open('./tests/resources/small_high_pass.pkl', 'rb') as file:
        arr = pickle.load(file)
    return arr


@pytest.fixture
def dummy_low_pass() -> np.ndarray:
    with open('./tests/resources/small_low_pass.pkl', 'rb') as file:
        arr = pickle.load(file)
    return arr


@pytest.fixture
def dummy_mask() -> np.ndarray:
    with open('./tests/resources/small_mask.pkl', 'rb') as file:
        arr = pickle.load(file)
    return arr


@pytest.fixture
def dummy_grain_object(dummy_grain_mask) -> Grain:
    grain = Grain(
        grain_id=0,
        grain_mask=dummy_grain_mask,
        grain_area=10.2,
        grain_circularity_rating=0.6
    )
    return grain


@pytest.fixture
def dummy_image_data_object(dummy_mask, dummy_high_pass, dummy_low_pass, dummy_original_image, dummy_grain_object, tmp_path) -> ImageData:
    image_data = ImageData(
        image_original=dummy_original_image,
        mask=dummy_mask,
        high_pass=dummy_high_pass,
        low_pass=dummy_low_pass,
        grains={0: dummy_grain_object},
        file_directory=tmp_path,
        filename="dummy_filename",
        mask_rgb=dummy_mask,
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
def dummy_perovstats_object(dummy_image_data_object, default_config) -> PerovStats:
    perovstats_object = PerovStats(
        images=[dummy_image_data_object],
        config=default_config
    )
    return perovstats_object

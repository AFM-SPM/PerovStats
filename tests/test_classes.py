import pytest

from perovstats.classes import Grain, ImageData


@pytest.mark.parametrize(
    ("expected"),
    [
        pytest.param(
            {
                "grain_id": 0,
                "grain_area": 10.2,
                "grain_circularity": 0.6
            }
        ),
    ]
)
def test_grain_to_dict(
        dummy_grain_object: Grain,
        expected: dict,
):
    grain_dict = dummy_grain_object.to_dict()

    assert grain_dict == expected


@pytest.mark.parametrize(
    ("expected"),
    [
        pytest.param(
            {
                'cutoff': 0.9,
                'cutoff_freq_nm': 1.32,
                'file_dir': 'tmp_path',
                'filename': 'dummy_filename',
                'grains_per_nm2': 2,
                'mask_area_nm': 100,
                'mask_size_x_nm': 10,
                'mask_size_y_nm': 10,
                'mean_grain_size_nm2': None,
                'median_grain_size_nm2': None,
                'mode_grain_size_nm2': None,
                'num_grains': 1,
                'pixel_to_nm_scaling': 1,
                'threshold': -0.8,
            }
        )
    ]
)
def test_image_data_to_dict(
        dummy_image_data_object: ImageData,
        expected: dict,
        tmp_path,
):
    expected["file_dir"] = tmp_path
    image_data_dict = dummy_image_data_object.to_dict()

    assert image_data_dict == expected

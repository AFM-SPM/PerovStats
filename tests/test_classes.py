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
        dummy_grain: Grain,
        expected: dict,
):
    grain_dict = dummy_grain.to_dict()

    assert grain_dict == expected


@pytest.mark.parametrize(
    ("expected"),
    [
        pytest.param(
            {
                'cutoff': 0.9,
                'cutoff_freq_nm': 1.32,
                'file_dir': 'file/dir',
                'filename': 'dummy_filename',
                'grains_per_nm2': 2,
                'mask_area_nm': 100,
                'mask_size_x_nm': 10,
                'mask_size_y_nm': 10,
                'num_grains': 1,
                'pixel_to_nm_scaling': 1,
            }
        )
    ]
)
def test_image_data_to_dict(
        dummy_image_data: ImageData,
        expected: dict,
):
    image_data_dict = dummy_image_data.to_dict()

    assert image_data_dict == expected

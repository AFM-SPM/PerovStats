import pytest

from perovstats.core.classes import Grain, ImageData


@pytest.mark.parametrize(
    ("expected"),
    [
        pytest.param(
            {
                "grain_id": 0,
                "grain_area": 10.2,
                "grain_circularity": 0.6,
                "grain_volume": None,
                "indented": False,
                'bbox_centre_x': 0,
                'bbox_centre_y': 0,
                'bbox_height_px': 10,
                'bbox_width_px': 10,
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
                'file_dir': 'tmp_path',
                'filename': 'dummy_filename',
                'grains_per_nm2': 2,
                'mask_area_nm': 100,
                'mask_size_x_nm': 10,
                'mask_size_y_nm': 10,
                'mean_grain_area_nm2': None,
                'median_grain_area_nm2': None,
                'mode_grain_area_nm2': None,
                'num_grains': 1,
                'pixel_to_nm_scaling': 1,
                'smears_removed': True,
                'smear_percent': 10.0,
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

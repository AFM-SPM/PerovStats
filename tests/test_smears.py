import numpy as np

from perovstats.smears import find_smear_areas, clean_smears


def test_find_smear_areas(
    default_config,
    dummy_image_data_object,
    dummy_smear_mask
):
    find_smear_areas(default_config, dummy_image_data_object)

    smear_mask = dummy_image_data_object.smears
    assert np.array_equal(smear_mask, dummy_smear_mask)


def test_clean_smears(dummy_high_pass, dummy_smear_mask):
    clean_smears(dummy_high_pass, dummy_smear_mask)

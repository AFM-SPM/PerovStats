import numpy as np

from perovstats.smears import find_smear_areas, clean_smears


def test_find_smear_areas(
    dummy_high_pass,
    dummy_low_pass,
    dummy_smear_mask,
    default_config
):
    config = default_config["remove_smears"]
    filename = "dummy_filename"
    smear_mask, _ = find_smear_areas(dummy_high_pass, dummy_low_pass, config, filename)

    assert np.array_equal(smear_mask, dummy_smear_mask)


def test_clean_smears(dummy_high_pass, dummy_smear_mask):
    clean_smears(dummy_high_pass, dummy_smear_mask)

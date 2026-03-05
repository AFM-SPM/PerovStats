import numpy as np

from perovstats.smears import find_smear_areas, get_horizontal_gradients


def test_find_smear_areas(
    dummy_high_pass,
    dummy_low_pass,
    dummy_smear_mask,
    default_config
):
    config = default_config["remove_smears"]
    filename = "dummy_filename"
    smear_mask, _, _ = find_smear_areas(dummy_high_pass, dummy_low_pass, config, filename)

    assert np.array_equal(smear_mask, dummy_smear_mask)


def test_get_horizontal_gradients(dummy_low_pass, dummy_low_pass_h_gradient):
    threshold = 50
    mask = get_horizontal_gradients(dummy_low_pass, threshold)

    assert np.array_equal(mask, dummy_low_pass_h_gradient)

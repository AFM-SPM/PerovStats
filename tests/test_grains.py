from perovstats.grains import tidy_border, find_grains

def test_find_grains(dummy_perovstats_object):
    find_grains(dummy_perovstats_object)

    assert dummy_perovstats_object.images[0].grains is not None


def test_tidy_borders(mask_random):
    mask = mask_random
    new_mask = tidy_border(mask)

    assert new_mask.shape == mask.shape
    assert new_mask.dtype == mask.dtype

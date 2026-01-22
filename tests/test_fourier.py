import pytest
import numpy as np

from perovstats.classes import PerovStats
from perovstats.fourier import split_frequencies, create_masks

def test_create_masks(dummy_perovstats_object: PerovStats, image_random):
    dummy_image = dummy_perovstats_object.images[0]
    dummy_image.image_original = image_random
    create_masks(dummy_perovstats_object)

    assert dummy_image.mask.shape == dummy_image.image_original.shape


def test_split_frequencies(dummy_perovstats_object: PerovStats, image_random):
    dummy_perovstats_object.images[0].image_original = image_random
    split_frequencies(dummy_perovstats_object)
    high_pass = dummy_perovstats_object.images[0].high_pass
    low_pass = dummy_perovstats_object.images[0].low_pass
    image = dummy_perovstats_object.images[0].image_original

    assert high_pass.shape == image.shape
    assert low_pass.shape == image.shape
    assert np.allclose((high_pass + low_pass), image)

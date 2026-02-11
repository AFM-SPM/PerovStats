import pytest
import pandas as pd
import numpy as np
from skimage.measure import regionprops, label

from perovstats.statistics import save_to_csv, save_config, find_circularity_rating

def test_save_config(tmp_path):
    data = {"config": "test"}
    out = tmp_path / "output.yaml"

    save_config(data, out)

    assert out.exists()


def test_save_to_csv(tmp_path):
    data = pd.DataFrame([{"test": "test"}])
    out = tmp_path / "output.csv"

    save_to_csv(data, out)

    assert out.exists()


@pytest.mark.parametrize(
        ("grain_mask", "expected_rating"),
        [
            pytest.param(
                np.array([
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                ]),
                0.9595052218168317,
                id="circle"
            ),
            pytest.param(
                np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                ]),
                0.43116456976745576,
                id="non-circle"
            ),
        ]
)
def test_find_circularity_rating(grain_mask, expected_rating):
    labeled = label(grain_mask)
    props = regionprops(labeled)[0]
    area = props.area
    perimeter = props.perimeter_crofton

    circularity_rating = find_circularity_rating(area, perimeter)

    assert circularity_rating == expected_rating

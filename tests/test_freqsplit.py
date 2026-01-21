import pytest
import numpy as np

from perovstats.freqsplit import extend_image, frequency_split

@pytest.mark.parametrize(
        ("image","expected"),
        [
            pytest.param(
                np.array(
                    [
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                    ]
                )
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]
                    ]
                )
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1],
                        [1, 0]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0]
                    ]
                )
            ),
            pytest.param(
                np.array([[1]]),
                np.array([[1]])
            ),
        ]

)
def test_extend_image(image, expected):
    extended_image, _ = extend_image(image)
    assert np.array_equal(extended_image, expected)


def test_create_frequency_mask():
    pass


def test_frequency_split():
    image = np.array(
        [
            [0, 1, 0.5],
            [0.6, 0, 0.3],
            [0.1, 0.9, 0.5]
        ]
    )
    cutoff = 0.01
    edge_width = 0.03

    expected_high_pass = np.array(
        [
            [-0.27252962, 0.72747038, 0.22747038],
            [0.32747038, -0.27252962, 0.02747038],
            [-0.17252962, 0.62747038, 0.22747038]
        ]
    )
    expected_low_pass = np.array(
        [
            [0.27252962, 0.27252962, 0.27252962],
            [0.27252962, 0.27252962, 0.27252962],
            [0.27252962, 0.27252962, 0.27252962]
        ]
    )

    high_pass, low_pass = frequency_split(image, cutoff, edge_width)

    print(high_pass)

    assert np.allclose(high_pass, expected_high_pass)
    assert np.allclose(low_pass, expected_low_pass)

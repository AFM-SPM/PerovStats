import pandas as pd

from perovstats.core.io import save_to_csv, save_config


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

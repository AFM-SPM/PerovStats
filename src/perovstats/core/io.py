from __future__ import annotations
from pathlib import Path
from yaml import safe_dump
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_image(image: np.ndarray, output_dir: Path, filename: str, cmap: str='grey') -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cmap:
        plt.imsave(output_dir / filename, image, cmap=cmap)
    else:
        plt.imsave(output_dir / filename, image)


def save_to_csv(df: pd.DataFrame, output_filename: str) -> None:
    """
    Method for saving pd.DataFrames to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The created dataframe to be saved.
    output_filename : str
        The output directory and filename.
    """
    df.to_csv(output_filename, index=False)


def save_config(config: dict, output_filename: str) -> None:
    """
    Method for saving config options to .yaml.

    Parameters
    ----------
    config : dict
        Dictionary containing all config options to be saved.
    output_filename: str
        The output directory and filename.
    """
    with (output_filename).open("w") as outfile:
        safe_dump(config, outfile, default_flow_style=False)

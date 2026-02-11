from yaml import safe_dump
import pandas as pd
import numpy as np


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


def find_circularity_rating(grain_area, grain_perimeter) -> float:
    """
    Take a grain mask and use the isoperimetric ratio to give it a rating (0 - 1)
    for how circular it is.
    """
    return (4 * np.pi * grain_area) / (grain_perimeter * grain_perimeter)

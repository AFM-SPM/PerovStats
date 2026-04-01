from __future__ import annotations
from pathlib import Path
from yaml import safe_dump
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_image(image: np.ndarray, output_dir: Path, filename: str, cmap: str='grey') -> None:
    """
    Save an array to file as an image

    Parameters
    ----------
    image : np.ndarray
        The array to save to file.
    output_dir : Path
        The directory to save the image to.
    filename : str
        The name of the file to be created/ saved to.
    cmap : str
        The cmap to be used in the imsave() function. By default is 'grey'.
    """
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


def grain_area_histogram(data, filename, output_dir):
    """
    Method for saving a histogram plotting the areas of grains found.

    Parameters
    ----------
    data : List[float]
        A list of the datapoints (areas) for each grain.
    filename : str
        The name of the .spm file being processed.
    output_dir : Path
        The main directory outputs are saved to.
    """
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data, bins='auto', kde=True, log_scale=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Grain areas nm²')
        plt.tight_layout()
        full_output_dir = output_dir / "graphs"
        full_output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_output_dir / f"{filename}_grain_areas_hist.png", dpi=300)
        plt.close(fig)


def grain_circularity_histogram(data, filename, output_dir):
    """
    Method for saving a histogram plotting the circularity rating of grains found.

    Parameters
    ----------
    data : List[float]
        A list of the datapoints (circularity rating) for each grain.
    filename : str
        The name of the .spm file being processed.
    output_dir : Path
        The main directory outputs are saved to.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, bins='auto', kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Grain circularities (0-1)')
    plt.tight_layout()
    full_output_dir = output_dir / "graphs"
    full_output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(full_output_dir / f"{filename}_grain_circularity_hist.png", dpi=300)
    plt.close(fig)

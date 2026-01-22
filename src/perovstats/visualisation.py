from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import skimage as ski
from loguru import logger
import numpy as np
import numpy.typing as npt
import seaborn as sns

CUTOFF_FREQ_NM = 250


def create_plots(
        filename: str,
        mask_areas: list,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        nm_to_micron: float,
):
    """Show plots for grain area distribution and the rgb image of identified grains"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Grain size distributions")
    plot_areas(mask_areas, nm_to_micron, title="grain area distribution", units="nm", ax=axes[0])
    plot_coloured_grains(filename, nm_to_micron, mask_data, ax=axes[1])
    plt.tight_layout()
    plt.show()


def plot_areas(areas: list, nm_to_micron: float, title: str | None = None, units: str = "um", ax=None) -> None:
    """Plot histogram of mask areas."""
    if ax is None:
        plt.gca()
    if title is None:
        title = ""
    if units == "um":
        areas = [area * nm_to_micron**2 for area in areas]
        ax.set_xlabel("area (µm²)")
    elif units == "nm":
        ax.set_xlabel("area (nm²)")
    else:
        msg = "units must be 'um' or 'nm'"
        raise ValueError(msg)
    sns.histplot(areas, kde=True, bins="auto", log_scale=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("count")


def plot_coloured_grains(
        filename: str,
        nm_to_micron: float,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        ax: int = None,
) -> None:
    """
    Plot coloured grains.

    Parameters
    ----------
    filename : str
        Name of the original .spm file
    nm_to_micron : float
        Scale factor of nm to microns.
    mask_data : dict[str, dict[str, npt.NDArray | float]]
        Dictionary containing an array of the mask to be coloured.
    ax : int
        The axis containing the coloured plot for the figure. (Improve this one).
    """
    if ax is None:
        ax = plt.gca()
    mask_rgb = mask_data["mask_rgb"]
    num_grains = mask_data["num_grains"]
    grains_per_nm2 = mask_data["grains_per_nm2"]
    grains_per_um2 = grains_per_nm2 / nm_to_micron**2
    mask_size_x_um = mask_data["mask_size_x_nm"] * nm_to_micron
    mask_size_y_um = mask_data["mask_size_y_nm"] * nm_to_micron
    title = (
        f"{filename}\n"
        f"image size: {mask_size_x_um} x {mask_size_y_um} µm² | "
        f"grains: {num_grains} | grains/µm²: {grains_per_um2:.2f}"
    )
    ax.imshow(mask_rgb)
    ax.set_title(title)
    ax.axis("off")

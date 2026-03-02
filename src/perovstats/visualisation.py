from __future__ import annotations

import matplotlib.pyplot as plt
import numpy.typing as npt
from pathlib import Path

import numpy as np

from .utils import normalise_array


def create_plots(
        output_dir: str,
        filename: str,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        nm_to_micron: float,
        image_object,
):
    """Show plots for grain area distribution and the rgb image of identified grains"""
    plot_coloured_grains(filename, nm_to_micron, mask_data, output_dir, image_object)


def plot_coloured_grains(
        filename: str,
        nm_to_micron: float,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        output_dir: str,
        image_object,
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
    mask_rgb = mask_data["mask_rgb"]
    # num_grains = mask_data["num_grains"]
    # grains_per_nm2 = mask_data["grains_per_nm2"]
    # grains_per_um2 = grains_per_nm2 / nm_to_micron**2
    # mask_size_x_um = mask_data["mask_size_x_nm"] * nm_to_micron
    # mask_size_y_um = mask_data["mask_size_y_nm"] * nm_to_micron
    # title = (
    #     f"{filename}\n"
    #     f"image size: {mask_size_x_um} x {mask_size_y_um} µm² | "
    #     f"grains: {num_grains} | grains/µm²: {grains_per_um2:.2f}"
    # )
    plot_name = filename + "_coloured_grains.jpg"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(Path(output_dir) / plot_name, mask_rgb)

    rgb_highpass = np.stack((image_object.high_pass,)*3, axis=-1)
    rgb_highpass = normalise_array(rgb_highpass)
    mask_2d = np.all(mask_rgb == 0, axis=2)
    rgb_highpass[mask_2d == 0] = [1, 1, 1]
    rgb_highpass[image_object.smears == 1] = [1, 0, 0]
    plt.imshow(rgb_highpass)
    plt.show()

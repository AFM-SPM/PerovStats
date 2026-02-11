from __future__ import annotations

import matplotlib.pyplot as plt
import numpy.typing as npt
from pathlib import Path


def create_plots(
        output_dir: str,
        filename: str,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        nm_to_micron: float,
):
    """Show plots for grain area distribution and the rgb image of identified grains"""
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_coloured_grains(filename, nm_to_micron, mask_data, ax=axes[1])
    plt.tight_layout()
    plot_name = filename + "_coloured_grains.jpg"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / plot_name, dpi=300, bbox_inches="tight")
    plt.close()


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

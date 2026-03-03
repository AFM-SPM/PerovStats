from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        imshows,
):
    """Show plots for grain area distribution and the rgb image of identified grains"""
    mask_rgb = mask_data["mask_rgb"]
    plot_name = filename + "_coloured_grains.jpg"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(Path(output_dir) / plot_name, mask_rgb)

    rgb_highpass = np.stack((image_object.high_pass,)*3, axis=-1)
    rgb_highpass = normalise_array(rgb_highpass)
    mask_2d = np.all(mask_rgb == 0, axis=2)
    rgb_highpass[mask_2d == 0] = [1, 1, 1]
    rgb_highpass[image_object.smears == 1] = [1, 0, 0]

    high_pass = image_object.high_pass
    mask_overlay = np.stack((high_pass,)*3, axis=-1)
    mask_overlay = normalise_array(mask_overlay)
    mask_overlay[image_object.mask > 0] = [1, 0, 0]

    # ~~~~ TEMP FOR VISUALISATION DURING DEBUGGING ~~~~~~~~~~~~~~~~~~~~~
    _, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(imshows[0], cmap="grey")
    axes[0, 0].set_title("Original high pass")
    axes[1, 0].imshow(imshows[1], cmap="grey")
    axes[1, 0].set_title("Combined masks")
    axes[0, 1].imshow(imshows[2], cmap="grey")
    axes[0, 1].set_title("High pass gradient comparison")
    axes[1, 1].imshow(imshows[3], cmap="grey")
    axes[1, 1].set_title("Low pass horizontal gradient")
    axes[0, 2].imshow(mask_overlay, cmap="grey")
    axes[0, 2].set_title("Grain edges")
    axes[1, 2].imshow(rgb_highpass, cmap="grey")
    axes[1, 2].set_title("Grains with smears removed")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # ~~~~ TEMP FOR VISUALISATION DURING DEBUGGING ~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    ax1.imshow(imshows[0], cmap="gray", interpolation="nearest")
    ax2.imshow(imshows[1], cmap="gray", interpolation="nearest")
    ax3.imshow(imshows[2], cmap="gray", interpolation="nearest")
    ax4.imshow(imshows[3], cmap="gray", interpolation="nearest")
    ax5.imshow(rgb_highpass, cmap="gray", interpolation="nearest")

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

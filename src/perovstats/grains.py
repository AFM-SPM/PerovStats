from __future__ import annotations
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from loguru import logger
from ruamel.yaml import YAML
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import regionprops

from .classes import Mask

LOGGER = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("./output")

# Annotated data
DATA_ANNOTATED = Path("path/to/perovskites_data_notated.csv")

NM_TO_MICRON = 1e-3

config_yaml_files = list(DATA_DIR.glob("*/**/*_config.yaml"))
logger.info(f"found {len(config_yaml_files)} config files")


def get_file_names() -> list[str]:
    files_to_include = []
    if DATA_ANNOTATED.exists():
        data_annotated = pd.read_csv(DATA_ANNOTATED)
        data_to_include = data_annotated[data_annotated.include == "Y"]
        files_to_include = data_to_include.filename.apply(Path, axis=1)
    names = [f.stem for f in files_to_include]
    return names


def plot_areas(areas: list, title: str | None = None, units: str = "um", ax=None) -> None:
    """Plot histogram of mask areas."""
    if ax is None:
        plt.gca()
    if title is None:
        title = ""
    title = title + f" n:{len(areas)}"
    if units == "um":
        areas = [area * NM_TO_MICRON**2 for area in areas]
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
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        col_num: int = 1,
        ax = None,
) -> None:
    """Plot coloured grains."""
    if ax is None:
        ax = plt.gca()
    # num_items = len(masks_data)
    # num_rows = (num_items + col_num - 1) // col_num
    # fig, ax = plt.subplots(num_rows, col_num, figsize=(6 * col_num, 6 * num_rows))
    # for i, (filename, mask_data) in enumerate(masks_data.items()):
    mask_rgb = mask_data["mask_rgb"]
    num_grains = mask_data["num_grains"]
    grains_per_nm2 = mask_data["grains_per_nm2"]
    grains_per_um2 = grains_per_nm2 / NM_TO_MICRON**2
    mask_size_x_um = mask_data["mask_size_x_nm"] * NM_TO_MICRON
    mask_size_y_um = mask_data["mask_size_y_nm"] * NM_TO_MICRON
    title = (
        f"{filename}\n"
        f"image size: {mask_size_x_um} x {mask_size_y_um} µm² | "
        f"grains: {num_grains} | grains/µm²: {grains_per_um2:.2f}"
    )
    # row = i // col_num
    # col = i % col_num
    # ax = np.atleast_2d(ax)
    # ax[row, col].imshow(mask_rgb, cmap="gray")
    # ax[row, col].set_title(title)
    ax.imshow(mask_rgb)
    ax.set_title(title)
    ax.axis("off")

def create_plots(masks: list[Mask], names: list[str] | None = None) -> None:
    all_masks_grain_areas = []
    all_masks_data = {}
    data = []

    for mask_object in masks:
        filename = mask_object.filename
        file_directory = mask_object.file_directory

        LOGGER.info(f"processing file {mask_object.filename:<50}")

        config_yaml = mask_object.config

        pixel_to_nm_scaling = config_yaml["pixel_to_nm_scaling"]

        mask_file = file_directory / f"{filename}_mask.npy"
        mask = np.load(mask_file).astype(bool)
        mask = np.invert(mask)

        labelled_mask = label(mask, connectivity=1)
        labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0)

        mask_regionprops = regionprops(labelled_mask)
        mask_areas = [
            regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
        ]
        all_masks_grain_areas.extend(mask_areas)

        mask_size_x_nm = mask.shape[1] * pixel_to_nm_scaling
        mask_size_y_nm = mask.shape[0] * pixel_to_nm_scaling
        mask_area_nm = mask_size_x_nm * mask_size_y_nm
        grains_per_nm2 = len(mask_areas) / mask_area_nm

        mask_data = {
            "mask_rgb": labelled_mask_rgb,
            "grains_per_nm2": grains_per_nm2,
            "mask_size_x_nm": mask_size_x_nm,
            "mask_size_y_nm": mask_size_y_nm,
            "mask_area_nm": mask_area_nm,
            "num_grains": len(mask_areas),
        }
        all_masks_data[f"{filename}-{config_yaml['cutoff_freq_nm']}"] = mask_data

        data.append(
            {
                "filename": filename,
                "grains_per_nm2": grains_per_nm2,
                "mask_size_x_nm": mask_size_x_nm,
                "mask_size_y_nm": mask_size_y_nm,
                "mask_area_nm": mask_area_nm,
                "num_grains": len(mask_areas),
                "dir": file_directory,
                "cutoff_freq_nm": config_yaml["cutoff_freq_nm"],
                "cutoff": config_yaml["cutoff"],
            },
        )

    if data:
        grain_stats = pd.DataFrame(data)

        logger.info(
            f"obtained {grain_stats['num_grains'].sum()} grains in {len(grain_stats)} masks",
        )

        # plot_areas(all_masks_grain_areas, title="all masks areas", units="nm")

        # plot_coloured_grains(all_masks_data, col_num=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        plot_areas(all_masks_grain_areas, title="all masks areas", units="nm", ax=axes[0])

        for i, (filename, mask_data) in enumerate(all_masks_data.items()):
            plot_coloured_grains(filename, mask_data, col_num=1, ax=axes[i + 1])

        plt.tight_layout()
        plt.show()

        # BROKEN / IDK WHAT IT'S ACTUALLY NEEDED FOR
        # sns.histplot(
        #     grain_stats,
        #     x="grains_per_nm2",
        #     hue="mask_size_x_nm",
        #     kde=False,
        #     bins="auto",
        #     log_scale=True,
        # )
        # plt.show()
    else:
        LOGGER.warning("No images to process in grains.")

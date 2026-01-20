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

from .classes import Mask, Grains, Grain
from .visualisation import create_plots

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


def find_grains(masks: list[Mask], names: list[str] | None = None) -> None:
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
        # grain_stats = pd.DataFrame(data)
        all_grains = {}
        for i, mask_area in enumerate(mask_areas):
            all_grains[i] = mask_area
        grain_stats = Grains(all_grains=all_grains, **data[0])

        logger.info(
            f"~~~ obtained {grain_stats.num_grains} grains from mask ~~~",
        )

        create_plots(filename, all_masks_grain_areas, all_masks_data, nm_to_micron=NM_TO_MICRON,)

        # Save grain data to .csv
    else:
        LOGGER.warning("No images to process in grains.")

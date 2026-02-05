from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy as np
import numpy.typing as npt
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import regionprops
from skimage import morphology

from .classes import Grain, PerovStats
from .visualisation import create_plots


# Data directory
DATA_DIR = Path("./output")

NM_TO_MICRON = 1e-3

config_yaml_files = list(DATA_DIR.glob("*/**/*_config.yaml"))
logger.info(f"found {len(config_yaml_files)} config files")


def find_grains(config, image_object, image_num) -> None:
    """
    Method to find grains from a mask and list the stats about them.

    Parameters
    ----------
    perovstats_object : PerovStats
        Class object containing all data from the process.

    Returns
    -------
    parovstats_object : PerovStats
        The updated class object.
    """
    all_masks_grain_areas = []
    data = []
    filename = image_object.filename
    config_yaml = config
    pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

    logger.info(f"processing file {image_object.filename:<50}")

    mask = image_object.mask.astype(bool)
    mask = np.invert(mask)
    labelled_mask = label(mask, connectivity=1)

    # Remove grains touching the edge
    labelled_mask = tidy_border(labelled_mask)
    labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0, saturation=0)

    mask_regionprops = regionprops(labelled_mask)
    mask_areas = [
        regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
    ]
    all_masks_grain_areas.extend(mask_areas)

    mask_size_x_nm = mask.shape[1] * pixel_to_nm_scaling
    mask_size_y_nm = mask.shape[0] * pixel_to_nm_scaling
    mask_area_nm = mask_size_x_nm * mask_size_y_nm
    grains_per_nm2 = len(mask_areas) / mask_area_nm

    if len(mask_areas) > 0:
        mean_grain_size = find_mean_grain_size(mask_areas)
        median_grain_size = find_median_grain_size(mask_areas)
        mode_grain_size = find_mode_grain_size(mask_areas)
    else:
        mean_grain_size = 0
        median_grain_size = 0
        mode_grain_size = 0

    mask_data = {
        "filename": filename,
        "mask_rgb": labelled_mask_rgb,
        "grains_per_nm2": grains_per_nm2,
        "mask_size_x_nm": mask_size_x_nm,
        "mask_size_y_nm": mask_size_y_nm,
        "mask_area_nm": mask_area_nm,
        "num_grains": len(mask_areas),
        "mean_grain_size": mean_grain_size,
        "median_grain_size": median_grain_size,
        "mode_grain_size": mode_grain_size
    }

    data.append(mask_data)


    # Assign area data for individual grains to appropriate classes
    for key, value in mask_data.items():
        setattr(image_object, key, value)
    image_object.grains = {}
    for i, grain_area in enumerate(mask_areas):
        image_object.grains[i] = Grain(grain_id=i, grain_area=grain_area)

    logger.info(
        f"[{filename}] obtained {image_object.num_grains} grains from mask {image_num}",
    )

    create_plots(Path(config_yaml["output_dir"]) / filename / "images", filename, mask_areas, mask_data, nm_to_micron=NM_TO_MICRON)


def find_median_grain_size(values):
    values = sorted(values)
    count = len(values)
    mid = count // 2

    if count % 2 == 1:
        return values[mid]
    else:
        return (values[mid - 1] + values[mid]) / 2


def find_mean_grain_size(values):
    return sum(values) / len(values)


def find_mode_grain_size(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


@staticmethod
def tidy_border(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Remove whole grains touching the border.

    Parameters
    ----------
    mask : npt.NDArray
        3-D Numpy array of the grain mask tensor.

    Returns
    -------
    npt.NDArray
        3-D Numpy array of the grain mask tensor with grains touching the border removed.
    """
    # Find the grains that touch the border then remove them from the full mask tensor
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    for region in mask_regionprops:
        if (
            region.bbox[0] == 0
            or region.bbox[1] == 0
            or region.bbox[2] == mask.shape[0]
            or region.bbox[3] == mask.shape[1]
        ):
            mask[mask_labelled == region.label] = 0

    return mask

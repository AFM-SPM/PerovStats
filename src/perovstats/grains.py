from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy as np
import numpy.typing as npt
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage import morphology

from .classes import Grain
from .visualisation import create_plots
from .statistics import find_circularity_rating
from .segmentation import create_grain_mask


# Data directory
DATA_DIR = Path("./output")

NM_TO_MICRON = 1e-3

config_yaml_files = list(DATA_DIR.glob("*/**/*_config.yaml"))
logger.info(f"found {len(config_yaml_files)} config files")


def find_grains(config, image_object, imshows) -> None:
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
    logger.info(f"[{image_object.filename}] : *** Grain finding ***")

    all_masks_grain_areas = []
    data = []
    filename = image_object.filename
    config_yaml = config
    pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

    mask = image_object.mask.astype(bool)
    mask = np.invert(mask)
    labelled_mask = label(mask, connectivity=1)

    # Remove grains touching the edge
    labelled_mask = tidy_border(labelled_mask)

    # Remove grains in/ touching smears
    if config["remove_smears"]["run"]:
        labelled_mask = clean_smears(labelled_mask, image_object.smears)

    labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0, saturation=0)

    mask_regionprops = regionprops(labelled_mask)
    mask_areas = [
        regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
    ]
    mask_perimeters = [
        regionprop.perimeter_crofton * pixel_to_nm_scaling for regionprop in mask_regionprops
    ]
    mask_images = [
        regionprop.image for regionprop in mask_regionprops
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
        grain_circularity = find_circularity_rating(grain_area, mask_perimeters[i])
        image_object.grains[i] = Grain(grain_id=i, grain_mask=mask_images[i], grain_area=grain_area, grain_circularity_rating=grain_circularity)

    logger.info(
        f"[{filename}] : Obtained {image_object.num_grains} grains",
    )

    create_plots(Path(config_yaml["output_dir"]) / filename / "images", filename, mask_data, image_object=image_object, imshows=imshows)


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


def clean_smears(mask: np.ndarray, smear_mask: np.ndarray) -> np.ndarray:
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    for region in mask_regionprops:
        region_mask = (mask_labelled == region.label)
        if np.any(region_mask & smear_mask.astype(bool)):
            mask[region_mask] = 0

    return mask


def find_threshold(
    filename: str,
    image: np.ndarray,
    threshold_func: callable,
    smooth_sigma: float,
    smooth_func,
    area_threshold,
    disk_radius,
    min_threshold,
    max_threshold,
):
    """
    Loop through possible threshold values and select the value
    that produces the most grains.

    Parameters
    ----------
    filename: str
        Name of the image being processed.
    image : np.ndarray
        Numpy array of the high-passed image to use.
    threshold_func : Callable
        Threshold function.
    smooth_sigma : Callable, optional
        Smoothing function.
    smooth_func : dict, optional
        Arguments to be passed to the smoothing function.
    area_threshold : float
        The area threshold.
    disk_radius : float
        The disk radius.
    pixel_to_nm_scaling : float
        The scale factor of pixels:nm.

    Returns
    -------
    float
        The selected best threshold.
    """
    logger.info(f"[{filename}] : Finding threshold")

    best_threshold = None
    best_grain_num = 0
    threshold_step = (max_threshold - min_threshold) / 50
    for curr_threshold in np.arange(min_threshold, max_threshold, threshold_step):
        curr_threshold = round(curr_threshold, 3)
        np_mask = create_grain_mask(
            image,
            threshold_func=threshold_func,
            threshold=curr_threshold,
            smooth_sigma=smooth_sigma,
            smooth_func=smooth_func,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
        )

        mask = np_mask.astype(bool)
        mask = np.invert(mask)

        labelled_mask = label(mask, connectivity=1)

        # Remove grains touching the edge
        labelled_mask = tidy_border(labelled_mask)
        mask_regionprops = regionprops(labelled_mask)

        if len(mask_regionprops) >= best_grain_num:
            best_grain_num = len(mask_regionprops)
            best_threshold = curr_threshold

    if best_grain_num == 0:
        logger.warning(f"[{filename}] : No grains could be found for any tested threshold.",
                       "consider increasing the threshold bounds in the config.",
                       "Skipping image..")
        return None

    logger.info(f"[{filename}] : Best threshold found: {best_threshold}")
    return best_threshold

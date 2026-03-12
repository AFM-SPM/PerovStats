from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

from .core.classes import Grain, ImageData
from .core.image_processing import normalise_array
from .core.segmentation import tidy_border
from .smears import clean_smears


def find_grains(
        config: dict[str, any],
        image_object: ImageData,
    ) -> None:
    """
    Method to find grains from a mask and list the stats about them.

    Parameters
    ----------
    perovstats_object : PerovStats
        Class object containing all data from the process.
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
        labelled_mask, area_removed = clean_smears(labelled_mask, image_object.smears)
    else:
        area_removed = 0
    smear_percent = round((area_removed / (mask.shape[0] * mask.shape[1])) * 100, 3)
    image_object.smear_percent = smear_percent

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

    # Save high-pass with mask skeleton overlay
    mask_rgb = mask_data["mask_rgb"]
    smear_overlay = np.stack((image_object.high_pass,)*3, axis=-1)
    smear_overlay = normalise_array(smear_overlay)
    mask_2d = np.all(mask_rgb == 0, axis=2)
    smear_overlay[mask_2d == 0] = [1, 1, 1]
    smear_overlay[image_object.smears == 1] = [1, 0, 0]
    save_dir = Path(config_yaml["output_dir"]) / filename / "images"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_dir / f"{filename}_smears.jpg", smear_overlay)


def find_median_grain_size(values: list[float]) -> float:
    values = sorted(values)
    count = len(values)
    mid = count // 2

    if count % 2 == 1:
        return values[mid]
    else:
        return (values[mid - 1] + values[mid]) / 2


def find_mean_grain_size(values: list[float]) -> float:
    return sum(values) / len(values)


def find_mode_grain_size(values: list[float]) -> float:
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def find_circularity_rating(grain_area: float, grain_perimeter: float) -> float:
    """
    Take a grain mask and use the isoperimetric ratio to give it a rating (0 - 1)
    for how circular it is.
    """
    return (4 * np.pi * grain_area) / (grain_perimeter * grain_perimeter)

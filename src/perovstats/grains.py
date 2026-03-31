from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage import morphology

from .core.classes import Grain, ImageData
from .core.image_processing import normalise_array
from .core.io import save_image
from .smears import clean_smears
from .statistics import grain_area_histogram, grain_circularity_histogram


def find_grains(
        config: dict[str, any],
        image_object: ImageData,
    ) -> None:
    """
    Method to find grains from a mask and list the stats about them.
    A mask is taken and segments are found before being filtered for things such as:
        - Grains with a size under a given threshold are removed
        - Grains touching the edge of the image are removed
        - Grains touching smear areas are removed
    Stats are then recorded for the grains, both individually and averages across
    the whole image.

    Parameters
    ----------
    config : dict[str, any]
        A dictionary of config options inputted at the start of the program.
    image_object : ImageData
        Dataclass reference contianing data and stats on the image currently
        being processed.
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
    min_dist_from_edge = 4
    labelled_mask = tidy_border(labelled_mask, min_dist_from_edge)

    max_size = 25
    labelled_mask = remove_small_grains(labelled_mask, max_size)

    # Remove grains in/ touching smears
    if config["remove_smears"]["run"]:
        labelled_mask = clean_smears(labelled_mask, image_object.smears)

    labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0, saturation=0)

    # Get the area, perimeter and individual grain images for each grain
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

    # Averages and overall stats for the entire image
    mask_size_x_nm = mask.shape[1] * pixel_to_nm_scaling
    mask_size_y_nm = mask.shape[0] * pixel_to_nm_scaling
    mask_area_nm = mask_size_x_nm * mask_size_y_nm
    grains_per_nm2 = len(mask_areas) / mask_area_nm

    if len(mask_areas) > 0:
        mean_grain_area = find_mean_grain_area(mask_areas)
        median_grain_area = find_median_grain_area(mask_areas)
        mode_grain_area = find_mode_grain_area(mask_areas)
    else:
        mean_grain_area = 0
        median_grain_area = 0
        mode_grain_area = 0

    mask_data = {
        "filename": filename,
        "mask_rgb": labelled_mask_rgb,
        "grains_per_nm2": grains_per_nm2,
        "mask_size_x_nm": mask_size_x_nm,
        "mask_size_y_nm": mask_size_y_nm,
        "mask_area_nm": mask_area_nm,
        "num_grains": len(mask_areas),
        "mean_grain_area": mean_grain_area,
        "median_grain_area": median_grain_area,
        "mode_grain_area": mode_grain_area
    }

    data.append(mask_data)


    # Assign area data for individual grains to appropriate classes
    for key, value in mask_data.items():
        setattr(image_object, key, value)
    image_object.grains = {}
    circularity_data = []
    for i, grain_area in enumerate(mask_areas):
        grain_circularity = find_circularity_rating(grain_area, mask_perimeters[i])
        circularity_data.append(grain_circularity)
        # grain_volume = find_grain_volume(mask, mask_regionprops[i], labelled_mask, mask_images[i], pixel_to_nm_scaling)
        image_object.grains[i] = Grain(
            grain_id=i,
            # centre_x
            # centre_y
            # is_intersected
            grain_mask=mask_images[i],
            grain_area=grain_area,
            grain_circularity_rating=grain_circularity,
            # grain_volume=grain_volume
        )

    logger.info(
        f"[{filename}] : Obtained {image_object.num_grains} grains",
    )

    # Save high-pass with mask skeleton overlay and an image colouring grains individually
    mask_rgb = mask_data["mask_rgb"]
    image_object.mask_rgb = mask_rgb
    save_dir = Path(config_yaml["output_dir"]) / filename / "images"
    save_image(mask_rgb, save_dir, f"{filename}_rgb_grains.jpg", cmap=None)
    smear_overlay = np.stack((image_object.high_pass,)*3, axis=-1)
    smear_overlay = normalise_array(smear_overlay)
    mask_2d = np.all(mask_rgb == 0, axis=2)
    smear_overlay[mask_2d == 0] = [1, 1, 1]
    smear_overlay[image_object.smears == 1] = [1, 0, 0]
    save_image(smear_overlay, save_dir, f"{filename}_smears.jpg")

    image_object.mask_areas = mask_areas
    image_object.circularity_data = circularity_data

    grain_area_histogram(mask_areas, filename, save_dir)
    grain_circularity_histogram(circularity_data, filename, save_dir)


def find_median_grain_area(values: list[float]) -> float:
    """
    Median value is found from an inputted list of values

    Parameters
    ----------
    values : list[float]
        List of areas of all the grains in an image.

    Returns
    -------
    float
        The median value from the given list.
    """
    values = sorted(values)
    count = len(values)
    mid = count // 2

    if count % 2 == 1:
        return values[mid]
    else:
        return (values[mid - 1] + values[mid]) / 2


def find_mean_grain_area(values: list[float]) -> float:
    """
    Mean value is found from an inputted list of values

    Parameters
    ----------
    values : list[float]
        List of areas of all the grains in an image.

    Returns
    -------
    float
        The mean value from the given list.
    """
    return sum(values) / len(values)


def find_mode_grain_area(values: list[float]) -> float:
    """
    Mode value is found from an inputted list of values

    Parameters
    ----------
    values : list[float]
        List of areas of all the grains in an image.

    Returns
    -------
    float
        The mode value from the given list.
    """
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def find_circularity_rating(grain_area: float, grain_perimeter: float) -> float:
    """
    Take a grain mask and use the isoperimetric ratio to give it a rating (0 - 1)
    for how circular it is. 0 would be a straight line and 1 would be a perfect
    circle.
    This uses the isoperimetric quotient, which uses the idea that a circle is the
    shape which has the maximum possible area for a given perimeter.

    Parameters
    ----------
    grain_area : float
        The area of the given grain.
    grain_perimeter : float
        The perimeter of the given grain.
    Returns
    -------
    float
        A value between 0-1 rating the inputted grain shape on how circular it is.
    """
    return (4 * np.pi * grain_area) / (grain_perimeter * grain_perimeter)


def remove_small_grains(mask, max_size):
    return morphology.remove_small_objects(mask.astype(int), max_size=max_size)


@staticmethod
def tidy_border(mask: np.ndarray[np.bool_], min_dist) -> np.ndarray[np.bool_]:
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
    # Find the grains that touch the border/ nearly touch the border then remove them from the full mask tensor
    # It is required to include grains almost touching the border here as the cellpose model can sometimes make straight grains
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    for region in mask_regionprops:
        if (
            region.bbox[0] < min_dist
            or region.bbox[1] < min_dist
            or region.bbox[2] > mask.shape[0] - min_dist
            or region.bbox[3] > mask.shape[1] - min_dist
        ):
            mask[mask_labelled == region.label] = 0

    return mask

# def find_grain_volume(mask: np.ndarray, mask_regionprop, labelled_mask: np.ndarray, grain_mask: np.ndarray, pixel_to_nm_scaling: float):
#     # Get mask of only the grain but the same shape as the entire mask
#     grain_only_mask = mask * (labelled_mask == mask_regionprop.label)
#     image_mask = np.ma.masked_array(mask, mask=np.invert(grain_only_mask), fill_value=np.nan).filled()

#     return np.nansum(image_mask) * pixel_to_nm_scaling**2 * (1e-9)**3

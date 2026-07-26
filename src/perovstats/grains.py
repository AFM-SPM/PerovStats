from __future__ import annotations
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage import morphology
from scipy.ndimage import binary_fill_holes
from scipy import stats

from .core.classes import Grain, ImageData
from .core.io import grain_area_histogram, grain_circularity_histogram
from .smears import clean_smears

MIN_DIST_FROM_EDGE = 4
MIN_GRAIN_SIZE = 25

def find_grains(
        config: dict[str, any],
        image_object: ImageData
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
        Dataclass reference containing data and stats on the image currently
        being processed.
    """
    logger.info(f"[{image_object.filename}] : *** Grain finding ***")

    filename = image_object.filename
    pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

    mask = image_object.mask.astype(bool)
    mask = np.invert(mask)
    labelled_mask = label(mask, connectivity=1)

    # Remove grains touching the edge
    labelled_mask, border_grains = tidy_border(labelled_mask, MIN_DIST_FROM_EDGE)
    image_object.edge_grains = border_grains

    # Remove grains not meeting minimum size requirements
    labelled_mask = remove_small_grains(labelled_mask, MIN_GRAIN_SIZE)

    # Remove grains in/ touching smear areas
    if config["remove_smears"]["run"]:
        labelled_mask, removed_mask = clean_smears(labelled_mask, image_object.smears)
        image_object.smear_grains = removed_mask

    # Remove grains with sizes or shapes outside n std devs of mean (n defined in config)
    if config["outliers"]["remove_outliers"]:
        labelled_mask, num_removed = remove_outliers(config, labelled_mask, pixel_to_nm_scaling, "area")
        logger.info(f"[{filename}] : {num_removed} grains considered outliers due to size and removed from the data.")

        labelled_mask, num_removed = remove_outliers(config, labelled_mask, pixel_to_nm_scaling, "shape")
        logger.info(f"[{filename}] : {num_removed} grains considered outliers due to shape and removed from the data.")
    else:
        logger.info(f"[{filename}] : Outlier removal is turned off in the config.")

    # Get the area, perimeter and individual grain images for each grain
    mask_regionprops = regionprops(labelled_mask)
    labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0, saturation=0)

    mask_details = _extract_regionprop_data(mask_regionprops, pixel_to_nm_scaling)

    # Get the image of the grain from the high-passed image
    mask_details['images'] = _extract_grain_images(image_object.high_pass, mask_regionprops)

    mask_data = _calculate_image_statistics(
        mask_shape=mask.shape,
        num_grains=len(mask_details['areas']),
        grain_areas=mask_details['areas'],
        pixel_to_nm_scaling=pixel_to_nm_scaling
    )
    mask_data['mask_rgb'] = labelled_mask_rgb

    # Assign area data for individual grains to appropriate classes
    for key, value in mask_data.items():
        setattr(image_object, key, value)

    # Create the class instances for each grain (and assign them to image_object.grains)
    _create_grain_objects(
        image_object=image_object,
        mask_details=mask_details
    )

    image_object.indent_mask = image_object.mask

    logger.info(
        f"[{filename}] : Obtained {image_object.num_grains} grains",
    )

    mask_rgb = mask_data["mask_rgb"]
    mask_rgb[image_object.indent_mask > 0] = [0, 0, 0]
    image_object.mask_rgb = mask_rgb

    # Save area and circularity data for all grains and export a histogram of them each
    save_dir = Path(config["output_dir"]) / filename / "images"
    image_object.mask_areas = mask_details['areas']
    image_object.circularity_data = mask_details['circularities']

    # Generate the histograms showing grain area and circularity rating
    font_size = config["output"]["font_size"]
    grain_area_histogram(mask_details['areas'], filename, save_dir, font_size)
    grain_circularity_histogram(mask_details['circularities'], filename, save_dir, font_size)


def _extract_regionprop_data(regionprops_list, scaling: float) -> dict:
    """
    Create a dictionary of lists, with each holding each grain's data regarding
    the category defined in the key.
    Some fields are stored in regionprops_list while others can be calculated using the pre-existing
    values.

    Parameters
    ----------
    regionprops_list
        List of dictionaries, one for each grain. The dictionaries contain basic information
        about a grain's shape/ location.
    scaling : float
        The pixel_to_nm_scaling.

    Returns
    -------
    dict[list]
        A dictionary with each key being a data category, with their values being lists
        with one element/ datapoint per grain.
    """
    return {
        'areas': [rp.area * scaling**2 for rp in regionprops_list],
        'perimeters': [rp.perimeter_crofton * scaling for rp in regionprops_list],
        'masks': [rp.image for rp in regionprops_list],
        'outlines': [_get_grain_outline(rp.image) for rp in regionprops_list],
        'bboxes': [rp.bbox for rp in regionprops_list],
        'circularities': [_find_circularity_rating(rp.area * scaling**2, rp.perimeter * scaling) for rp in regionprops_list]
    }


def calculate_grain_statistics(grain_areas: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate mean, median, and mode for grain areas.

    Parameters
    ----------
    grain_areas : np.ndarray
        An array of areas for all grains detected.

    Returns
    -------
    tuple[float, float, float]
        A tuple consisting of three floats:
            - mean area
            - median_area
            - mode_area.
    """
    if len(grain_areas) == 0:
        return 0.0, 0.0, 0.0

    mean_area = np.mean(grain_areas)
    median_area = np.median(grain_areas)

    # Use scipy.stats.mode for proper handling
    mode_result = stats.mode(grain_areas, keepdims=True)
    mode_area = mode_result.mode[0]

    return mean_area, median_area, mode_area


def _extract_grain_images(
    high_pass_image: np.ndarray,
    regionprops_list: list
) -> list[np.ndarray]:
    """
    Extract individual grain images from high-pass filtered image.

    Parameters
    ----------
    high_pass_image : np.ndarray
        The highpassed version of the original image.
    regionprops_list : list[dict[str, any]]
        A list of basic data about each grain.

    Return
    ------
    list[np.ndarray]
        List of 2d arrays, one for each grain.
    """
    grain_images = []
    for regionprop in regionprops_list:
        bbox_slice = regionprop.slice
        hollow_mask = regionprop.image
        filled_mask = binary_fill_holes(hollow_mask)
        crop = high_pass_image[bbox_slice]
        grain_image = np.where(filled_mask, crop, 0)
        grain_images.append(grain_image)
    return grain_images


def _calculate_image_statistics(
    mask_shape: tuple[int, int],
    num_grains: int,
    grain_areas: np.ndarray,
    pixel_to_nm_scaling: float
) -> dict[str, float]:
    """
    Calculate overall image statistics.

    Parameters
    ----------
    mask_shape : tuple[int, int]
        Two ints giving the width and length of the iamge in pixels.
    num_grains : int
        Number of grains detected in the image.
    grain_areas : np.ndarray
        An array of areas for all grains detected.
    pixel_to_nm_scaling : float
        The pixel to nm scaling of the image.

    Returns
    -------
    dict[str, any]
        A dictionary of stats collected about the image as a whole.
    """
    height, width = mask_shape
    mask_size_x_nm = width * pixel_to_nm_scaling
    mask_size_y_nm = height * pixel_to_nm_scaling
    mask_area_nm = mask_size_x_nm * mask_size_y_nm
    grains_per_nm2 = num_grains / mask_area_nm if mask_area_nm > 0 else 0

    mean_area, median_area, mode_area = calculate_grain_statistics(grain_areas)

    return {
        "grains_per_nm2": grains_per_nm2,
        "mask_size_x_nm": mask_size_x_nm,
        "mask_size_y_nm": mask_size_y_nm,
        "mask_area_nm": mask_area_nm,
        "num_grains": num_grains,
        "mean_grain_area": mean_area,
        "median_grain_area": median_area,
        "mode_grain_area": mode_area,
    }


def _create_grain_objects(
    image_object: ImageData,
    mask_details: dict[list[str, any]]
) -> None:
    """
    Create Grain objects and return circularity data.

    Parameters
    ----------
    image_object : ImageData
        The ImageData class instance for the current image.
    mask_details : dict[list[str, any]]
        A dictionary containing data categories, each holding a list with
        one value per grain.
    """
    circularity_data = []
    image_object.grains = {}

    areas = mask_details["areas"]
    perimeters = mask_details["perimeters"]
    images = mask_details["images"]
    masks = mask_details["masks"]
    outlines = mask_details["outlines"]
    bboxes = mask_details["bboxes"]

    for i, (area, perimeter, image, mask, outline, bbox) in enumerate(
        zip(areas, perimeters, images, masks, outlines, bboxes)
    ):
        # Calculate and collect additional data using the existing data categories
        circularity = _find_circularity_rating(area, perimeter)
        circularity_data.append(circularity)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        centre_x = round(bbox[0] + (0.5 * width))
        centre_y = round(bbox[1] + (0.5 * height))

        # Create a new Grain dataclass instance and add it to an empty
        # index in image_object.grains
        image_object.grains[i] = Grain(
            grain_id=i,
            grain_image=image,
            grain_mask=mask,
            grain_mask_outline=outline,
            grain_area=area,
            grain_circularity_rating=circularity,
            grain_bbox=bbox,
            grain_size_px=(width, height),
            grain_centre_coords=(centre_x, centre_y),
        )


def _find_circularity_rating(grain_area: float, grain_perimeter: float) -> float:
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
    return max(0.0, min((4 * np.pi * grain_area) / (grain_perimeter * grain_perimeter), 1.0))


def remove_small_grains(mask, max_size):
    return morphology.remove_small_objects(mask.astype(int), max_size=max_size)


def tidy_border(mask: np.ndarray[np.bool_], min_dist: float) -> np.ndarray[np.bool_]:
    """
    Remove whole grains touching the border. Also save data on the mask lines that contain a grain
    touching the edge of the image, to be removed later.

    Parameters
    ----------
    mask : npt.NDArray
        3-D Numpy array of the grain mask tensor.
    min_dist: float
        Minimum distance from border a grain can be to not be removed.

    Returns
    -------
    npt.NDArray
        3-D Numpy array of the grain mask tensor with grains touching the border removed.
    """
    # Find the grains that touch the border/ nearly touch the border then remove them from the full mask tensor
    # It is required to include grains almost touching the border here as the cellpose model can sometimes make straight grains
    # 1 or 2 pixel away from and parallel to an edge of the image so we want to make sure these are removed.
    removed_mask = np.zeros_like(mask, dtype=bool)
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    to_remove = []
    keep_indices = []
    for region in mask_regionprops:
        # If grain region is too close too/ touching an edge
        if (
            region.bbox[0] < min_dist
            or region.bbox[1] < min_dist
            or region.bbox[2] > mask.shape[0] - min_dist
            or region.bbox[3] > mask.shape[1] - min_dist
        ):
        # Assign to one of two lists depending on the if statement
            to_remove.append(region)
        else:
            keep_indices.append(region.label)

    # Create a mask of just non-removed grains
    kept_mask = np.isin(mask_labelled, keep_indices)
    # No fly zone is the kept_mask with a 1px border around all sections
    # This is so the skeletonised mask around these grains is included in this zone
    # so it's not deleted during cleaning.
    no_fly_zone = morphology.dilation(kept_mask, footprint=morphology.disk(1))

    # All grain masks/ outlines of grains to be removed are deleted
    # However mask outlines touching a kept and a removed grain are still kept
    # to avoid kept grains missing outline
    for region in to_remove:
            grain_pixels = (mask_labelled == region.label)
            dilated_grain = morphology.dilation(grain_pixels, footprint=morphology.disk(1))
            # Exclusive-or, meaning all pixels that're True in both arrays are discounted
            # as well as all pixels that're False in both arrays.
            # This leaves us with just the dilated pixels.
            outer_halo = dilated_grain ^ grain_pixels
            # safe_halo is anywhere this dilated outline is and the no fly zone is False.
            safe_halo = outer_halo & ~no_fly_zone
            removed_mask[safe_halo] = True
            mask[grain_pixels] = 0

    return mask, removed_mask


def _get_grain_outline(mask: np.ndarray) -> np.ndarray:
    """
    Get a mask of a single-pixel outline of a given grain.
    The grain is extended by 1px in each direction and then boundaries are found from this.

    Parameters
    ----------
    mask : np.ndarray
        A filled mask of the grain.

    Returns
    -------
    np.ndarray
        The input mask with the centre unfilled, leaving just a 1px wide outline.
    """
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    boundary = find_boundaries(padded_mask, mode='inner')

    return boundary[1:-1, 1:-1]


def remove_outliers(
    config: dict[str, Any],
    labelled_mask: np.ndarray,
    pixel_to_nm_scaling: float,
    criteria: str = "area"
) -> tuple[np.ndarray, int]:
    """
    Remove grains that are statistical outliers.

    Parameters
    ----------
    criteria : str
        Either "area" for grain size or "shape" for circularity.

    Returns
    -------
    np.ndarray
        The updated labelled_mask after outliers have been removed.
    int
        The number of grains marked as outliers and removed.
    """
    mask_regionprops = regionprops(labelled_mask)

    # Collect the required data
    if criteria == "area":
        values = np.array([rp.area * pixel_to_nm_scaling**2 for rp in mask_regionprops])
    elif criteria == "shape":
        areas = np.array([rp.area * pixel_to_nm_scaling**2 for rp in mask_regionprops])
        perimeters = np.array([rp.perimeter_crofton * pixel_to_nm_scaling for rp in mask_regionprops])
        values = np.array([_find_circularity_rating(a, p) for a, p in zip(areas, perimeters)])
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    z_scores = np.abs(stats.zscore(values))
    outliers = z_scores > config["outliers"]["max_z_score"]

    num_removed = 0
    for i, region in enumerate(mask_regionprops):
        if outliers[i]:
            labelled_mask[labelled_mask == region.label] = 0
            num_removed += 1

    return labelled_mask, num_removed

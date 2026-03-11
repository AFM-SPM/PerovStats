import numpy as np
from scipy.ndimage import binary_closing, label, binary_dilation
from loguru import logger
from scipy import ndimage as ndi
from skimage import morphology
from skimage.measure import regionprops

from .core.image_processing import get_horizontal_gradients


def find_smear_areas(
        high_pass: np.ndarray,
        low_pass: np.ndarray,
        config: dict[str, any],
        filename: str,
    ):
    """
    Take a high-passed array and for each pixel compare its gradient difference on each axis
    with the horizontal axis' gradient for the low-passed image's corresponding pixel, pixels
    over given thresholds for both masks are marked as smear areas.

    Parameters
    ----------
    high_pass: np.ndarray
        The high-passed version of the image for use in finding vertical gradients significantly
        bigger than the corresponding horizontal gradient.
    low_pass: np.ndarray
        The low-passed version of the image for use in finding high horizontal gradients.
    config: dict[str, any]
        Smear removal configuration options
    filename: str
        Filename currently being processed, used for logging info.
    """
    threshold = config["smear_threshold"]
    smooth_sigma= config["smooth_sigma"]
    min_size = config["min_smear_area"]
    lowpass_threshold = config["lowpass_threshold"]

    MIN_SMEAR_AREAS = 6

    logger.info(f"[{filename}] : *** Finding smear areas ***")

    # Compare the horizontal and vertical gradient of the high pass, marking pixels(/ areas) that have a
    # value over a given threshold
    smooth = ndi.gaussian_filter(high_pass, sigma=smooth_sigma)
    grad_x = np.abs(ndi.sobel(smooth, axis=1))
    grad_y = np.abs(ndi.sobel(smooth, axis=0))

    stripe_score = grad_y / (grad_x + 1e-6) # 1e-6 prevents 0 division

    mask = stripe_score > threshold

    # Remove smears not meeting minimum area requirements
    labeled, n = label(mask)
    for i in range(1, n+1):
        if np.sum(labeled == i) < min_size:
            mask[labeled == i] = 0

    mask = binary_closing(mask, structure=np.ones((5, 10)))

    # Compare the mask calculated above with a mask selecting all pixels with a horizontal gradient over
    # a given threshold, creating a new mask containing all overlapping pixels
    low_pass_gradient_mask = get_horizontal_gradients(low_pass, threshold=lowpass_threshold)
    final_mask = mask & low_pass_gradient_mask
    final_mask = binary_dilation(final_mask, structure=np.ones((3, 3)))

    # Remove smears not meeting minimum area requirements
    labeled, n = label(final_mask)
    for i in range(1, n+1):
        if np.sum(labeled == i) < min_size:
            final_mask[labeled == i] = 0

    _, n = label(final_mask)

    logger.info(f"[{filename}] : Smear areas found: {n}")
    if n < MIN_SMEAR_AREAS:
        logger.info(f"[{filename}] : Minimum number of smear areas not met, skipping smear removal.")
        empty_mask = np.zeros_like(final_mask)
        return empty_mask, False

    return final_mask, True


def clean_smears(mask: np.ndarray, smear_mask: np.ndarray) -> np.ndarray:
    """
    Compare the found grain segments with the previously computed smear mask
    and remove grains that overlap with any part of the mask.

    Parameters
    ----------
    mask : np.ndarray
        The grain mask.
    smear_mask : np.ndarray
        The smear mask.

    Returns
    -------
    np.ndarray
        The resultant grain mask with sections overlapping with the smear mask removed.

    """
    removed_grains_area = 0
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    for region in mask_regionprops:
        region_crop = mask_labelled[region.slice] == region.label
        smear_crop = smear_mask[region.slice].astype(bool)
        if np.any(region_crop & smear_crop):
            removed_grains_area += region.area
            mask[region.slice][region_crop] = 0

    return mask, removed_grains_area

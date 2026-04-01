import numpy as np
from scipy.ndimage import binary_closing, label, binary_dilation
from loguru import logger
from scipy import ndimage as ndi
from skimage import morphology
from skimage.measure import regionprops

from .core.classes import ImageData
from .core.image_processing import get_horizontal_gradients


def find_smear_areas(
        config: dict[str, any],
        image_object: ImageData,
    ) -> None:
    """
    Smears are errors in an AFM scan image and should be ignored/ removed from the data.
    They are characteristed by areas of horizontal lines, most visible in the high-passed
    version of an image and occur when the gradient of the overall material is too high at
    a given point.
    This method takes a high-passed array and for each pixel compares its gradient difference on each axis
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
    config = config["remove_smears"]
    if config["run"]:
        threshold = config["smear_threshold"]
        smooth_sigma= config["smooth_sigma"]
        min_size = config["min_smear_size"]
        min_smear_area_percent = config["min_smear_area_percent"]
        lowpass_threshold = config["lowpass_threshold"]

        high_pass = image_object.high_pass
        low_pass = image_object.low_pass
        filename = image_object.filename

        logger.info(f"[{filename}] : *** Finding smear areas ***")

        # Compare the horizontal and vertical gradient of the high pass, marking pixels(/ areas) that have a
        # value over a given threshold
        smooth = ndi.gaussian_filter(high_pass, sigma=smooth_sigma)
        grad_x = np.abs(ndi.sobel(smooth, axis=1))
        grad_y = np.abs(ndi.sobel(smooth, axis=0))

        # Value given to each pixel based on the difference between horizontal and vertical gradient
        stripe_score = grad_y / (grad_x + 1e-6) # 1e-6 prevents 0 division

        mask = stripe_score > threshold

        # Remove smears not meeting minimum area requirements
        labeled, n = label(mask)
        for i in range(1, n+1):
            if np.sum(labeled == i) < min_size:
                mask[labeled == i] = 0

        mask = binary_closing(mask, structure=np.ones((5, 10)))

        # Compare the mask calculated above with a mask selecting all pixels with a horizontal gradient over
        # a given threshold in the low-pass image, creating a new mask containing all overlapping pixels
        low_pass_gradient_mask = get_horizontal_gradients(low_pass, threshold=lowpass_threshold)
        final_mask = mask & low_pass_gradient_mask
        final_mask = binary_dilation(final_mask, structure=np.ones((3, 3)))

        # Remove smears not meeting minimum area requirements
        labeled, n = label(final_mask)
        for i in range(1, n+1):
            if np.sum(labeled == i) < min_size:
                final_mask[labeled == i] = 0

        _, n = label(final_mask)

        percentage = round(np.mean(final_mask) * 100, 2)

        logger.info(f"[{filename}] : Smear areas found: {n} ({percentage}% of mask)")
        if percentage < min_smear_area_percent:
            logger.info(f"[{filename}] : Minimum smear coverage not met, skipping smear removal.")
            final_mask = np.zeros_like(final_mask)
            smears_removed = False
            percentage = 0
        else:
            smears_removed = True

        image_object.smears = final_mask
        image_object.smears_removed = smears_removed
        image_object.smear_percent = percentage
    else:
        image_object.smears = np.zeros_like(image_object.high_pass)
        image_object.smears_removed = False
        image_object.smear_percent = 0


def clean_smears(mask: np.ndarray, smear_mask: np.ndarray) -> np.ndarray:
    """
    Compare the found grain segments with the previously computed smear mask
    and remove grains that overlap with any part of the mask.
    Also keep a log of

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
    removed_mask = np.zeros_like(mask, dtype=bool)
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)

    keep_labels = []
    remove_regions = []

    no_fly_zone = morphology.dilation(mask, footprint=morphology.disk(1))

    # for each grain in mask
    for region in mask_regionprops:
        region_crop = mask_labelled[region.slice] == region.label
        smear_crop = smear_mask[region.slice].astype(bool)

        # if any pixels overlap with smear mask, remove them
        if np.any(region_crop & smear_crop):
            remove_regions.append(region)
        else:
            keep_labels.append(region.label)

    good_grains_mask = np.isin(mask_labelled, keep_labels)
    no_fly_zone = morphology.dilation(good_grains_mask, footprint=morphology.disk(1))

    for region in remove_regions:
        grain_pixels = (mask_labelled == region.label)

        # Get the mask pixels bordering this grain
        dilated_grain = morphology.dilation(grain_pixels, footprint=morphology.disk(1))
        outer_halo = dilated_grain ^ grain_pixels

        # Only keep this mask border where it's not touching a border of an existing grain
        safe_halo = outer_halo & ~no_fly_zone

        # Add the border sections safe to delete to the removed_mask and remove the grain
        removed_mask[safe_halo] = True
        mask[grain_pixels] = 0

    return mask, removed_mask

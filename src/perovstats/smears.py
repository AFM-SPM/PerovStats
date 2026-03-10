import numpy as np
from scipy.ndimage import binary_closing, label, binary_dilation
from loguru import logger
from scipy import ndimage as ndi


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

    labeled, n = label(final_mask)

    logger.info(f"[{filename}] : Smear areas found: {n}")
    if n < MIN_SMEAR_AREAS:
        logger.info(f"[{filename}] : Minimum number of smear areas not met, skipping smear removal.")
        empty_mask = np.zeros_like(final_mask)
        return empty_mask, False

    return final_mask, True


def get_horizontal_gradients(
        image: np.ndarray,
        threshold: float
    ) -> np.ndarray:
    """
    Use the sobel formula to assign a gradient to each pixel in an array
    for the horizontal axis.
    """
    grad_x = ndi.sobel(image, axis=1)
    mask = grad_x > threshold

    return mask

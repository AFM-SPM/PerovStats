import numpy as np
from scipy.ndimage import binary_closing, label, binary_dilation
from loguru import logger
from scipy import ndimage as ndi


def find_smear_areas(
        high_pass: np.ndarray,
        low_pass: np.ndarray,
        config,
        filename,
    ):
    threshold = config["smear_threshold"]
    smooth_sigma= config["smooth_sigma"]
    min_size = config["min_smear_area"]
    lowpass_threshold = config["lowpass_threshold"]

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

    imshows = [high_pass, final_mask, mask, low_pass_gradient_mask]

    return final_mask, imshows


def get_horizontal_gradients(image, threshold):
    grad_x = ndi.sobel(image, axis=1)
    mask = grad_x > threshold

    return mask

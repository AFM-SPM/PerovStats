import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_closing, label
from loguru import logger
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from .utils import normalise_array


def find_smear_areas(
        image: np.ndarray,
        low_pass: np.ndarray,
        config,
        filename,
        sigma=1,
        threshold=2,
        min_size=200
    ):
    smooth = ndi.gaussian_filter(image, sigma=sigma)
    grad_x = np.abs(ndi.sobel(smooth, axis=1))
    grad_y = np.abs(ndi.sobel(smooth, axis=0))

    stripe_score = grad_y / (grad_x + 1e-6) # 1e-6 prevents 0 division

    mask = stripe_score > threshold

    labeled, n = label(mask)
    for i in range(1, n+1):
        if np.sum(labeled == i) < min_size:
            mask[labeled == i] = 0

    mask = binary_closing(mask, structure=np.ones((5, 10)))

    low_pass_gradient_mask = get_low_pass_gradients(low_pass, threshold=100)

    final_mask = mask & low_pass_gradient_mask

    _, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes[0, 0].imshow(image, cmap="grey")
    axes[0, 1].imshow(final_mask, cmap="grey")
    axes[1, 0].imshow(mask, cmap="grey")
    axes[1, 1].imshow(low_pass_gradient_mask, cmap="grey")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return final_mask


def get_low_pass_gradients(image, threshold):
    grad_x = ndi.sobel(image, axis=1)
    mask = grad_x > threshold

    return mask


def find_smear_areas_og(image: np.ndarray, config: dict[str, any], filename: str):
    """
    Remove given areas from the image by setting their height values as -1
    so the rest of the program knows to ignore them.

    Parameters
    ----------
    image : np.ndarray
        The input image to detect and remove smears from.

    Returns
    -------
    np.ndarray
        Edited version of the input image with smear areas all set to -1 height.
    """

    logger.info(f"[{filename}] : *** Smear cleaning ***")

    smear_threshold = config["threshold"]
    min_smear_length = config["min_smear_length"]
    difference_sensitivity = config["difference_sensitivity"]
    min_neighbours = config["min_neighbours"]
    min_smear_area = config["min_smear_area"]
    connection_distance = config["connection_distance"]
    neighbour_range = 9 # How far to check for mask neighbours vertically, must be odd

    image = normalise_array(image)
    mask = np.zeros_like(image, dtype=bool)
    for y in range(image.shape[0]-1):
        x = 0
        while x < image.shape[1]:
            # Check if y+1 or y-1 are significantly different, if not skip
            if get_pixel_difference(image[y,x], image[y-1,x]) > smear_threshold or get_pixel_difference(image[y,x], image[y+1,x]) > smear_threshold:
                up_diff = get_pixel_difference(image[y,x],image[y-1,x])
                down_diff = get_pixel_difference(image[y,x],image[y+1,x])
                x_start = x
                curr_x = x
                line_end = False
                while not line_end:
                    curr_x += 1
                    if curr_x < image.shape[0]:
                        curr_up_diff = get_pixel_difference(image[y,curr_x],image[y-1,curr_x])
                        curr_down_diff = get_pixel_difference(image[y,curr_x],image[y+1,curr_x])
                        if abs(curr_up_diff - up_diff) < difference_sensitivity or abs(curr_down_diff - down_diff) < difference_sensitivity:
                            pass
                        else:
                            line_end = True
                            if curr_x - x_start > min_smear_length:
                                for x_pos in range(x_start, curr_x):
                                    mask[y, x_pos] = True
                    else:
                        line_end = True
                x = curr_x
            x += 1

    # Remove mask pixels if it doesn't have enough vertical neighbours
    filtered_mask = np.zeros_like(image, dtype=bool)
    coords = np.argwhere(mask)
    radius = neighbour_range // 2
    for coord in coords:
        row_start = max(coord[0] - radius, 0)
        row_end = min(coord[0] + radius + 1, mask.shape[0])
        vertical_slice = mask[row_start:row_end, coord[1]]
        num_true = np.sum(vertical_slice)
        has_neighbours = num_true > min_neighbours
        if has_neighbours:
            filtered_mask[tuple(coord)] = True

    # Connect fragmented regions
    structure = np.ones((connection_distance,connection_distance), dtype=bool)
    filtered_mask = binary_closing(filtered_mask, structure=structure)

    # Find regions and ignore small ones
    labelled, num_features = scipy.ndimage.label(filtered_mask)
    sizes = scipy.ndimage.sum(filtered_mask, labelled, range(1, num_features + 1))
    large_labels = np.where(sizes > min_smear_area)[0] + 1
    filtered_mask = np.isin(labelled, large_labels)

    logger.info(f"[{filename}] : Smear mask created")

    return filtered_mask


def get_pixel_difference(pixel1, pixel2):
    return abs(pixel1 - pixel2)

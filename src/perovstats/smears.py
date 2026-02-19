import numpy as np
import scipy.ndimage
from scipy.ndimage import binary_closing

from .utils import normalise_array

def create_smear_mask(image: np.ndarray):
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
    import matplotlib.pyplot as plt

    smear_areas = find_smear_areas(image)
    image[smear_areas] = -25 # temporary while testing

    return smear_areas


def find_smear_areas(image: np.ndarray):
    smear_threshold = 0.09 # Difference between vertical neighbours to trigger a possible smear start
    min_smear_length = 10 # Minimum length of a smear row
    difference_sensitivity = 0.03 # Maximum difference between x and x+1's difference to their vertical neighbours
    neighbour_range = 9 # How far to check for mask neighbours vertically, must be odd
    min_neighbours = 2 # Minimum number of vertical neighbours found to keep the mask pixel
    min_smear_size = 100 # Minimum area of a smear
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
    structure = np.ones((7,7), dtype=bool)
    filtered_mask = binary_closing(filtered_mask, structure=structure)

    # Find regions and ignore small ones
    labelled, num_features = scipy.ndimage.label(filtered_mask)
    sizes = scipy.ndimage.sum(filtered_mask, labelled, range(1, num_features + 1))
    large_labels = np.where(sizes > min_smear_size)[0] + 1
    filtered_mask = np.isin(labelled, large_labels)

    return filtered_mask


def get_pixel_difference(pixel1, pixel2):
    return abs(pixel1 - pixel2)

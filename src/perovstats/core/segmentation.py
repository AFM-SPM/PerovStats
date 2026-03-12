from __future__ import annotations

import numpy.typing as npt
from skimage import morphology
from skimage.measure import regionprops
from scipy.special import erf
import skimage as ski
import numpy as np


def clean_mask(
    mask: np.ndarray,
    area_threshold: float = 100,
    disk_radius: int = 4,
) -> np.ndarray:
    """
    Clean up grain mask by connecting close segments and removing small sections.

    Parameters
    ----------
    mask : np.ndarray
        Mask array.
    area_threshold : float, optional
        Area threshold for cleaning up mask.
    disk_radius : int, optional
        Disk radius for cleaning up mask.

    Returns
    -------
    numpy.ndarray
        Cleaned up mask array.
    """
    mask = ski.morphology.remove_small_holes(
        ski.morphology.remove_small_objects(mask, max_size=area_threshold)
    )
    return ski.morphology.opening(mask, ski.morphology.disk(disk_radius))


def create_grain_mask(
    im: np.ndarray,
    threshold_block_size: float,
    smooth_sigma: float,
    area_threshold: float,
    disk_radius: float,
) -> np.ndarray:
    """
    Use local thresholding to find grain edges and create a skeletonised mask of
    borders.

    Parameters
    ----------
    im : np.ndarray
        Image to be masked.
    threshold_block_size : float
        Size of blocks to be thresholded once at a time.
    smooth_sigma : float
        Amount of smoothing applied to the image before thresholding.
    area_threshold : float
        Maximum size of a grain considered too small to count.
    disk_radius : float
        How far to look for closeby segments when connecting them.

    Returns
    -------
    np.ndarray
        Skeletonised mask of grain borders.
    """
    im_ = ski.filters.gaussian(im, sigma=smooth_sigma) # Smooth image
    local_thresh = ski.filters.threshold_local(im_, block_size=threshold_block_size) # Gets an array of local thresholds
    mask = im_ > local_thresh
    mask = clean_mask(mask, area_threshold, disk_radius) if area_threshold else mask
    selection = ski.util.invert(mask)
    return ski.morphology.skeletonize(selection)


def create_frequency_mask(
    f_grid,
    cutoff: float,
    edge_width: float = 0,
) -> np.ndarray:
    """
    Create a mask to filter frequencies.

    Parameters
    ----------
    f_grid : np.ndarray
        Frequency grid of the image, each pixel containing a value indicating
        the distance from the zero-frequency component in a radial frequency map.
    cutoff : float
        The spatial frequency cut off, expressed as a relative fraction
        of the Nyquist frequency.
    edge_width : float
        Edge width, expressed as a relative fraction of the Nyquist
        frequency. If zero, the filter has sharp edges. For non-zero
        values the transition has the shape of the error function,
        with the specified width.

    Returns
    -------
    np.ndarray
        Frequency mask.
    """
    if edge_width:
        return 0.5 * (erf((f_grid - cutoff) / edge_width) + 1)
    else:
        return (f_grid >= cutoff).astype(np.float64)


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

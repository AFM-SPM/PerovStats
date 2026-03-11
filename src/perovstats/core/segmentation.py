from __future__ import annotations
from typing import TYPE_CHECKING

from loguru import logger
import numpy.typing as npt
from skimage import morphology
from skimage.measure import regionprops, label
from scipy.special import erf
import skimage as ski

if TYPE_CHECKING:
    from collections.abc import Callable
import numpy as np


def threshold_mean_std(im: np.ndarray, k: float = 4) -> float:
    """
    Global mean/std dev threshold.

    Parameters
    ----------
    im : np.ndarray
        Image array.
    k : float
        Value of parameter `k` in threshold formula.

    Returns
    -------
    float
        Threshold value.
    """
    return im.mean() + k * im.std()


def threshold_mad(im: np.ndarray, k: float = 4) -> float:
    """
    Global median + median absolute deviance threshold.

    Parameters
    ----------
    im : np.ndarray
        Image array.
    k : float
        Value of parameter `k` in threshold formula.

    Returns
    -------
    float
        Threshold value.
    """
    med = np.median(im)
    mad = np.median(np.abs(im.astype(np.float32) - med))
    return med + mad * k * 1.4826


def clean_mask(
    mask: np.ndarray,
    area_threshold: float = 100,
    disk_radius: int = 4,
) -> np.ndarray:
    """
    Clean up grain mask.

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
    threshold_func: Callable = threshold_mean_std,
    threshold: float | None = None,
    smooth_func: Callable | None = None,
    smooth_sigma: float | None = None,
    area_threshold: float | None = None,
    disk_radius: float | None = None,
) -> np.ndarray:
    """
    Create a grain mask based on the specified threshold method.

    Create a grain mask based on the specified threshold function,
    optionally smoothing the input image before thresholding.

    Parameters
    ----------
    im : numpy.ndarray
        The image to be masked.
    pixel_to_nm_scaling : float
        The scale of the image for standardising parameters
    threshold_func : Callable
        Threshold function.
    threshold : float
        Threshold value.
    threshold_args : dict, optional
        Arguments to be passed to the threshold function.
    smooth : Callable, optional
        Smoothing function.
    smooth_args : dict, optional
        Arguments to be passed to the smoothing function.
    clean : Callable, optional
        Mask cleaning function.
    clean_args : dict, optional
        Arguments to be passed to the cleaning function.

    Returns
    -------
    numpy.ndarray
        Mask array.
    """

    im_ = smooth_func(im, sigma=smooth_sigma) if smooth_func else im
    mask = im > threshold_func(im_, k=threshold)
    mask = clean_mask(mask, area_threshold, disk_radius) if area_threshold else mask
    selection = ski.util.invert(mask)
    return ski.morphology.skeletonize(selection)


def create_grain_mask_2(
    im: np.ndarray,
    threshold_func: Callable = threshold_mean_std,
    threshold: float | None = None,
    smooth_func: Callable | None = None,
    smooth_sigma: float | None = None,
    area_threshold: float | None = None,
    disk_radius: float | None = None,
) -> np.ndarray:
    im_ = ski.filters.gaussian(im, sigma=smooth_sigma) # Smooth image
    threshold_block_size = 55

    local_thresh = ski.filters.threshold_local(im_, block_size=threshold_block_size) # Gets an array of local thresholds
    mask = im_ > local_thresh
    mask = clean_mask(mask, area_threshold, disk_radius) if area_threshold else mask
    selection = ski.util.invert(mask)
    return ski.morphology.skeletonize(selection)


def create_frequency_mask(
    shape: tuple[int, int],
    cutoff: float,
    edge_width: float = 0,
) -> np.ndarray:
    """
    Create a mask to filter frequencies.

    Parameters
    ----------
    shape : tuple
        Shape of the image to be masked.
    cutoff : float
        The spatial frequency cut off, expressed as a relative
        fraction of the Nyquist frequency.
    edge_width : float
        Edge width, expressed as a relative fraction of the Nyquist
        frequency.  If zero, the filter has sharp edges.  For non-zero
        values the transition has the shape of the error function,
        with the specified width.

    Returns
    -------
    np.ndarray
        Frequency mask.
    """
    yres, xres = shape
    xr = np.arange(xres)
    yr = np.arange(yres)
    fx = 2 * np.fmin(xr, xres - xr) / xres
    fy = 2 * np.fmin(yr, yres - yr) / yres

    # full coordinate arrays
    xx, yy = np.meshgrid(fx, fy)
    f = np.sqrt(xx**2 + yy**2)

    return (
        0.5 * (erf((f - cutoff) / edge_width) + 1)
        if edge_width
        else np.where(f >= cutoff, 1, 0)
    )


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


def find_threshold(
    filename: str,
    image: np.ndarray,
    pixel_to_nm_scaling: float,
    threshold_func: callable,
    smooth_sigma: float,
    smooth_func: callable,
    area_threshold: float,
    disk_radius: int,
    min_threshold: float,
    max_threshold: float,
) -> float:
    """
    Loop through possible threshold values and select the value
    that produces the most grains.

    Parameters
    ----------
    filename: str
        Name of the image being processed.
    image : np.ndarray
        Numpy array of the high-passed image to use.
    pixel_to_nm_scaling : float
        Scale of the image for parameter standardisation.
    threshold_func : Callable
        Threshold function.
    smooth_sigma : Callable, optional
        Smoothing function.
    smooth_func : dict, optional
        Arguments to be passed to the smoothing function.
    area_threshold : float
        The area threshold.
    disk_radius : float
        The disk radius.
    pixel_to_nm_scaling : float
        The scale factor of pixels:nm.

    Returns
    -------
    float
        The selected best threshold.
    """
    logger.info(f"[{filename}] : Finding threshold")

    best_threshold = None
    best_grain_num = 0
    threshold_step = (max_threshold - min_threshold) / 50
    for curr_threshold in np.arange(min_threshold, max_threshold, threshold_step):
        curr_threshold = round(curr_threshold, 3)
        np_mask = create_grain_mask(
            image,
            threshold_func=threshold_func,
            threshold=curr_threshold,
            smooth_sigma=smooth_sigma,
            smooth_func=smooth_func,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
        )

        mask = np_mask.astype(bool)
        mask = np.invert(mask)

        labelled_mask = label(mask, connectivity=1)

        # Remove grains touching the edge
        labelled_mask = tidy_border(labelled_mask)
        mask_regionprops = regionprops(labelled_mask)

        if len(mask_regionprops) >= best_grain_num:
            best_grain_num = len(mask_regionprops)
            best_threshold = curr_threshold

    if best_grain_num == 0:
        logger.warning(f"[{filename}] : No grains could be found for any tested threshold.",
                       "consider increasing the threshold bounds in the config.",
                       "Skipping image..")
        return None

    logger.info(f"[{filename}] : Best threshold found: {best_threshold}")
    return best_threshold

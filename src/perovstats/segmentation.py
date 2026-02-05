from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
import numpy as np
import skimage as ski


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

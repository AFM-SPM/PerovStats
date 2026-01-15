#
# Copyright: © 2025 University of Sheffield
#
# Authors:
#   Tamora James <t.d.james@sheffield.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Classical segmentation for PerovStats."""

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
        ski.morphology.remove_small_objects(mask, min_size=area_threshold),
        area_threshold=area_threshold,
    )
    return ski.morphology.opening(mask, ski.morphology.disk(disk_radius))


def create_grain_mask(
    im: np.ndarray,
    threshold: Callable = threshold_mean_std,
    threshold_args: dict | None = None,
    smooth: Callable | None = None,
    smooth_args: dict | None = None,
    clean: Callable | None = None,
    clean_args: dict | None = None,
) -> np.ndarray:
    """
    Create a grain mask based on the specified threshold method.

    Create a grain mask based on the specified threshold function,
    optionally smoothing the input image before thresholding.

    Parameters
    ----------
    im : numpy.ndarray
        The image to be masked.
    threshold : Callable
        Threshold function.
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
    smooth_args = smooth_args if smooth_args else {}
    im_ = smooth(im, **smooth_args) if smooth else im
    threshold_args = threshold_args if threshold_args else {}
    mask = im > threshold(im_, **threshold_args)
    clean_args = clean_args if clean_args else {}
    mask = clean(mask, **clean_args) if clean else mask
    selection = ski.util.invert(mask)
    return ski.morphology.skeletonize(selection)

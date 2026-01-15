# Copyright: © 2025 University of Sheffield
#
# Authors:
#   Tamora James <t.d.james@sheffield.ac.uk>
#   Toby Allwood <t.allwood@sheffield.ac.uk>
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
"""Frequency splitting for PerovStats."""

from __future__ import annotations
import cv2
import numpy as np
import pyfftw
from scipy.special import erf


def extend_image(
    image: np.ndarray,
    method: int = cv2.BORDER_REFLECT,
) -> tuple[np.ndarray, dict]:
    """
    Extend image by specified method.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be extended.
    method : int, optional
        Border type as specified in cv2.

    Returns
    -------
    tuple
        The extended image and a dictionary specifying the size of the borders.

    Raises
    ------
    NotImplementedError
        If `method` is not `cv2.BORDER_REFLECT`.
    """
    if method != cv2.BORDER_REFLECT:
        msg = f"Method {method} not implemented"
        raise NotImplementedError(msg)

    rows, cols = image.shape
    v_ext = rows // 2
    h_ext = cols // 2
    extent = {"top": v_ext, "bottom": v_ext, "left": h_ext, "right": h_ext}

    # Extend the image by mirroring to avoid edge effects
    extended_image = cv2.copyMakeBorder(
        image,
        **extent,
        borderType=method,
    )

    return extended_image, extent


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


def frequency_split(
    image: np.ndarray,
    cutoff: float,
    edge_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform frequency split on the specified image.

    Parameters
    ----------
    image : np.ndarray
        Image to be split.
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
    tuple
        High pass and low pass filtered images.
    """
    # Extend the image by mirroring to avoid edge effects
    extended_image, extent = extend_image(image, method=cv2.BORDER_REFLECT)

    shape = extended_image.shape

    # Set up FFTW objects
    fft_input = pyfftw.empty_aligned(shape, dtype="complex128")
    ifft_input = pyfftw.empty_aligned(shape, dtype="complex128")

    fft_object = pyfftw.builders.fft2(fft_input)
    ifft_object = pyfftw.builders.ifft2(ifft_input)

    # Apply DFT to extended image
    fft_input[:] = extended_image
    dft = fft_object()

    # Create mask to filter to specified frequencies
    mask = create_frequency_mask(extended_image.shape, cutoff, edge_width=edge_width)

    # Mask the DFT output
    dft = dft * mask

    # Perform reverse FFT on masked image to get high frequency content
    ifft_input[:] = dft
    high_pass = np.real(ifft_object())

    # Crop back to the original image size
    high_pass = high_pass[
        extent["top"] : -extent["bottom"],
        extent["left"] : -extent["right"],
    ]

    return high_pass, image - high_pass

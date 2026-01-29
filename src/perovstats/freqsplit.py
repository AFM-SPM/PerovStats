from __future__ import annotations
import cv2

import numpy as np
import pyfftw
from scipy import ndimage
from scipy.special import erf
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

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


def frequency_split_old(
    image: np.ndarray,
    cutoff: float,
    edge_width: float,
    pixel_to_nm_scaling: float,
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

    # cutoff = 2 * pixel_to_nm_scaling / cutoff

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


def frequency_split(
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a given image into two separate images, one containing a smooth background
    and the other containing the smaller details.

    Parameters
    ----------
    image : np.ndarray
        The image to process.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays: The details extracted from the image, and the
        remaining background after extracting the details.
    """
    extended_image, extent = extend_image(image, method=cv2.BORDER_REFLECT)
    pattern = extract_details(extended_image)

    pattern = pattern[
        extent["top"] : -extent["bottom"],
        extent["left"] : -extent["right"],
    ]

    background = image - pattern

    return pattern, background


def find_optimal_blur_sigma(image: np.ndarray):
    """
    Find the sigma value that removes the most detail compared to the
    previous value, for use as the selected cutoff point when splitting the images.

    1) Create a list of possible sigmas and loop through, using each for a gaussian blur.
    2) Measure strength of remaining detail for each with np.std(detail)
    3) Calculate the gradient between each detail strength value and find the biggest rate of
       change between values, using this as the selected sigma for the gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        The image to blur.

    Returns
    -------
    int
        The optimal sigma value found.
    """
    sigmas = np.linspace(1, 50, 20)
    energy_loss = []

    for s in sigmas:
        blurred = gaussian_filter(image.astype(float), sigma=s)
        detail = image - blurred
        energy_loss.append(np.std(detail))

    energy_loss = np.array(energy_loss)
    gradients = np.gradient(energy_loss)
    curvature = np.gradient(gradients)

    optimal_idx = np.argmax(curvature)
    return sigmas[optimal_idx]


def extract_details(image: np.ndarray):
    """
    Blur the image with the appropriate sigma and take the difference between
    this and the original as the perovskite pattern.
    Uses two passes of a gaussian filter.

    Parameters
    ----------
    image : np.ndarray
        The image to blur.

    Returns
    -------
    np.ndarray
        An array containing the details extracted from the image.
    """
    best_sigma = find_optimal_blur_sigma(image)

    first_background = gaussian_filter(image.astype(float), sigma=best_sigma)
    first_pattern = image - first_background

    leakage = gaussian_filter(first_pattern, sigma=best_sigma)

    refined_background = first_background + leakage
    pattern = image - refined_background

    # Increate contrast and normalise the pattern layers
    # pattern = increase_contrast_linear(refined_pattern)
    pattern = normalise_pattern(pattern)

    return pattern


def normalise_pattern(pattern_layer):
    # Centering the pattern at 0 and scaling so 1 unit = 1 standard deviation
    p_mean = np.mean(pattern_layer)
    p_std = np.std(pattern_layer)

    # This ensures that 'height' is relative to the texture's own roughness
    return (pattern_layer - p_mean) / (p_std + 1e-8)


def increase_contrast_linear(layer, percentiles=(2, 98)):
    p_low, p_high = np.percentile(layer, percentiles)
    # Clip the data to these percentiles to remove outlier spikes
    stretched = np.clip(layer, p_low, p_high)
    # Stretch to 0-1
    return (stretched - p_low) / (p_high - p_low)

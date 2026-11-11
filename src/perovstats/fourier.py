from __future__ import annotations

import cv2
import numpy as np
import pyfftw
from loguru import logger
from scipy.optimize import bisect
from scipy.special import erf

from .core.classes import ImageData
from .core.image_processing import extend_image


def split_frequencies(
    config: dict[str, any],
    image_object: ImageData,
    split_freq_nm: float,
) -> None:
    """
    Carry out frequency splitting on an image.

    All frequency parameters are expressed in real-world units, so the same
    config gives the same physical split regardless of the image's pixel size:

    - ``cutoff_bounds`` : [min, max] cutoff wavelength in nm.
    - ``edge_width``    : width of the filter transition in cycles per nm (nm^-1).

    Parameters
    ----------
    config : dict[str, any]
        Dictionary of all configuration options.
    image_object: ImageData
        Class object of ImageData, containing all data connected
        to the current image.
    split_freq_nm : float
        Cutoff wavelength in nm.
    """
    # Bring required config options into variables
    freqsplit_config = config["fourier"]
    edge_width = freqsplit_config["edge_width"]  # nm^-1
    min_cutoff_nm = freqsplit_config["cutoff_bounds"][0]
    max_cutoff_nm = freqsplit_config["cutoff_bounds"][1]
    min_rms = freqsplit_config["min_rms"]
    cutoff_nm = freqsplit_config["cutoff_nm"]

    filename = image_object.filename
    if image_object.image_flattened is not None:
        image = image_object.image_flattened
    else:
        image = image_object.image_original

    if freqsplit_config["run"]:
        # Size of one pixel in nm
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

        # Shortest wavelength the image can represent along an axis
        nyquist_nm = 2 * pixel_to_nm_scaling

        logger.info(f"[{filename}] : *** Frequency splitting ***")

        # Skip cutoff detection if a wavelength has already been selected (via notebook sliders or previous image)
        if split_freq_nm is None:
            # Keep cutoff as is if it was defined in config
            if cutoff_nm is None:
                cutoff_nm = find_cutoff(
                    image_object,
                    edge_width=edge_width,
                    min_cutoff_nm=min_cutoff_nm,
                    max_cutoff_nm=max_cutoff_nm,
                    min_rms=min_rms,
                    pixel_to_nm_scaling=pixel_to_nm_scaling,
                )

                # The next stages will require a cutoff value, so the rest of the processing for the current image has to
                # be skipped if this fails
                if not cutoff_nm:
                    logger.error(f"[{filename}] : Cutoff wavelength could not be determined, try adjusting the cutoff_bounds parameter. Skipping image..")
                    image_object.success = False
                    return
        else:
            cutoff_nm = split_freq_nm

        if cutoff_nm < nyquist_nm:
            logger.warning(
                f"[{filename}] : Cutoff wavelength ({round(cutoff_nm, 4)}nm) is below the smallest wavelength this "
                f"image can resolve ({round(nyquist_nm, 4)}nm), so the high-pass image will be almost empty. "
                "Check that cutoff_bounds is given in nm."
            )

        # Equivalent cutoff as a fraction of the Nyquist frequency. Not used by the filter, kept for reference
        cutoff = nyquist_nm / cutoff_nm
        logger.info(f"[{filename}] : Frequency cutoff: {round(cutoff_nm, 4)}nm ({round(cutoff, 4)} of Nyquist)")

        # Update image class with chosen cutoff
        image_object.cutoff = cutoff
        image_object.cutoff_freq_nm = cutoff_nm

        logger.info(f"[{filename}] : Splitting image frequencies")
        # Perform the fourier transformation using the found cutoff
        high_pass, low_pass = perform_fourier(
            image,
            cutoff_nm=cutoff_nm,
            edge_width=edge_width,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
        )

        image_object.high_pass = high_pass
        image_object.low_pass = low_pass
    else:
        # If frequency splitting is disabled we just assign the whole original image to image_object.high_pass as this variable
        # is used through the rest of the program. image_object.low_pass is filled with 0s
        logger.info(f"[{image_object.filename}] : Frequency splitting is disabled by config, the original image will be used.")
        if image_object.image_flattened is not None:
            image_object.high_pass = image_object.image_flattened
            image_object.low_pass = np.zeros_like(image_object.high_pass)
        else:
            image_object.high_pass = image_object.image_original
            image_object.low_pass = np.zeros_like(image_object.high_pass)


def perform_fourier(
    image: np.ndarray,
    cutoff_nm: float,
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

    freq_grid = create_frequency_mask(extended_image, pixel_to_nm_scaling)

    # Create mask to filter to specified frequencies
    mask = apply_cutoff(freq_grid, cutoff_nm, edge_width=edge_width)

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


def find_cutoff(
        image_object: ImageData,
        edge_width: float,
        min_cutoff_nm: float,
        max_cutoff_nm: float,
        min_rms: float,
        pixel_to_nm_scaling: float,
) -> float:
    """
    Find an ideal cutoff wavelength for performing a fourier transform on
    the current image.
    A fourier transform is setup and executed, and a binary search algorithm
    is used to test various cutoffs within the given range to find a value
    closest to the ideal RMS value given.

    Parameters
    ----------
    image_object : ImageData
        Class object of the current image containing all data
        connected to it.
    edge_width : float
        Width of the filter transition in cycles per nm (nm^-1). If zero,
        the filter has sharp edges. For non-zero values the transition has
        the shape of the error function, with the specified width.
    min_cutoff_nm : float
        Shortest cutoff wavelength to consider, in nm.
    max_cutoff_nm : float
        Longest cutoff wavelength to consider, in nm.
    min_rms : float
        RMS threshold for when checking cutoffs.
    pixel_to_nm_scaling : float
        Size of one pixel in nm.

    Returns
    -------
    float
        The best cutoff wavelength found, in nm.
    """
    logger.info(f"[{image_object.filename}] : Finding cutoff frequency...")
    image = image_object.image_original
    best_cutoff_nm = None

    # Set the min and max cutoffs (nm) based on real world scan size
    if min_cutoff_nm is None:
        min_cutoff_nm = 2 * pixel_to_nm_scaling # Smallest detail the image can hold
    if max_cutoff_nm is None:
        max_cutoff_nm = max(image.shape) * pixel_to_nm_scaling # Width of the scan in nm

    # Perform a fourier transform to find
    extended_image, _ = extend_image(image, method=cv2.BORDER_REFLECT)
    # Align memory to store data on the fft as efficiently as possible
    fft_input = pyfftw.empty_aligned(extended_image.shape, dtype="complex128")
    # Analysed computer hardware to find the more efficient way of performing the fourier transform
    fft_object = pyfftw.builders.fft2(fft_input)

    # Copies data into the allocated memory and performs the transform
    fft_input[:] = extended_image
    full_dft = fft_object()

    n_pixels = full_dft.size

    # Scale the minimum RMS with the pixel:nm scaling value
    min_rms = min_rms + (pixel_to_nm_scaling * 0.1)

    # Define vars used for the binary search
    freq_grid = create_frequency_mask(extended_image, pixel_to_nm_scaling)
    power_spectrum_flat = np.abs(full_dft).ravel()**2
    freq_grid_flat = freq_grid.ravel()
    n_pixels_sq = n_pixels**2

    # The function to take a cutoff wavelength and determine an rms from to be called multiple times by `bisect()`.
    # A longer cutoff wavelength lets more of the spectrum into the high pass, so the rms rises with cutoff_nm
    def objective(cutoff_nm):
        mask = apply_cutoff(freq_grid_flat, cutoff_nm, edge_width=edge_width)
        current_sum_sq = np.dot(power_spectrum_flat, mask)
        current_rms = np.sqrt(current_sum_sq / n_pixels_sq)
        return current_rms - min_rms

    try:
        # Run `objective()` multiple times to find the value giving an rms closest to the defined min_rms.
        # xtol is in nm
        best_cutoff_nm = bisect(objective, min_cutoff_nm, max_cutoff_nm, xtol=1e-3)
        logger.info(f"[{image_object.filename}] : Cutoff selected: {best_cutoff_nm:.4f}nm")
        return best_cutoff_nm
    except ValueError:
        logger.warning(
            f"[{image_object.filename}] : No cutoff could be found for the image. Please consider adjusting the cutoff bounds in the config."
        )
        return None


def apply_cutoff(
    f_grid,
    cutoff_nm: float,
    edge_width: float = 0,
) -> np.ndarray:
    """
    Create a mask to filter frequencies. Using the found cutoff frequency.

    Parameters
    ----------
    f_grid : np.ndarray
        Frequency grid of the image, each pixel containing a value indicating
        the distance from the zero-frequency component in a radial frequency map.
    cutoff_nm : float
        The cutoff wavelength in nm. Frequencies above 1 / cutoff_nm
        are kept by the mask.
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
    if cutoff_nm <= 0:
        raise ValueError(f"Cutoff wavelength must be positive, got {cutoff_nm}nm.")

    # Convert the wavelength to a spatial frequency in cycles per nm
    cutoff_freq = 1.0 / cutoff_nm

    if edge_width and edge_width > 0:
        return 0.5 * (erf((f_grid - cutoff_freq) / edge_width) + 1)
    else:
        return (f_grid >= cutoff_freq).astype(np.float64)


def create_frequency_mask(image: np.ndarray, pixel_to_nm_scaling: float) -> np.ndarray:
    """
    Create a 2D grid of normalised spatial frequencies for an image.
    Calculate the distance of each pixel from the zero frequency component
    in the Fourier domain.

    Parameters
    ----------
    image :np.ndarray
        2D image to be analysed.

    Returns
    -------
    np.ndarray
        A 2D arrya of the same shape as the input image containing the radial
        normalised frequencies of sqrt(fx^2 + fy^2).
    """
    # Create frequency mask grid
    yres, xres = image.shape

    # xr = np.arange(xres)
    # yr = np.arange(yres)
    # fx = 2 * np.fmin(xr, xres - xr) / xres
    # fy = 2 * np.fmin(yr, yres - yr) / yres

    fx = np.abs(np.fft.fftfreq(xres, d=pixel_to_nm_scaling))
    fy = np.abs(np.fft.fftfreq(yres, d=pixel_to_nm_scaling))

    # Full coordinate arrays
    xx, yy = np.meshgrid(fx, fy)
    return np.sqrt(xx**2 + yy**2)


def remove_extremes(high_pass: np.ndarray):
    """
    Clip an array to remove the highest and lowest extremes.

    Parameters
    ----------
    high_pass : np.ndarray
        The high-passed image to clip.

    Returns
    -------
    np.ndarray
        The clipped high-passed image.
    """
    p_low, p_high = np.percentile(high_pass, [1, 99])
    clipped = np.clip(high_pass, p_low, p_high)
    return clipped

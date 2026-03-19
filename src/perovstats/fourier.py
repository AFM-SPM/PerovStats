from __future__ import annotations
from pathlib import Path
import cv2

from loguru import logger
import numpy as np
import pyfftw
from scipy.special import erf

from .core.classes import ImageData
from .core.image_processing import extend_image, normalise_array
from .core.segmentation import create_frequency_mask
from .core.io import save_image


def split_frequencies(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    """
    Carry out frequency splitting on an image.

    Parameters
    ----------
    config : dict[str, any]
        Dictionary of all configuration options.
    image_object: ImageData
        Class object of ImageData, containing all data connected
        to the current image.

    Raises
    ------
    ValueError
        If neither `cutoff` nor `cutoff_freq_nm` argument supplied.
    """
    freqsplit_config = config["freqsplit"]
    edge_width = freqsplit_config["edge_width"]
    min_cutoff = freqsplit_config["cutoff_bounds"][0]
    max_cutoff = freqsplit_config["cutoff_bounds"][1]
    cutoff_step = freqsplit_config["cutoff_step"]
    min_rms = freqsplit_config["min_rms"]
    output_dir = Path(config["output_dir"])

    filename = image_object.filename
    file_output_dir = Path(output_dir / filename)
    file_output_dir.mkdir(parents=True, exist_ok=True)
    if image_object.image_flattened is not None:
        image = image_object.image_flattened
    else:
        image = image_object.image_original

    if freqsplit_config["run"]:
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

        logger.info(f"[{filename}] : *** Frequency splitting ***")

        cutoff = find_cutoff(
            image_object,
            edge_width,
            min_cutoff=min_cutoff,
            max_cutoff=max_cutoff,
            cutoff_step=cutoff_step,
            min_rms=min_rms,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
        )

        if not cutoff:
            logger.error(f"[{filename}] : Cutoff frequency could not be determined. Skipping image..")
            return

        cutoff_nm = 2 * pixel_to_nm_scaling / cutoff
        logger.info(f"[{image_object.filename}] : Frequency cutoff: {cutoff} ({np.round(cutoff_nm, 4)}nm)")

        # Update image class with chosen cutoff
        image_object.cutoff = cutoff
        image_object.cutoff_freq_nm = cutoff_nm

        logger.info(f"[{filename}] : Splitting image frequencies")
        high_pass, low_pass = perform_fourier(
            image,
            cutoff=cutoff,
            edge_width=edge_width,
        )

        image_object.high_pass = high_pass
        image_object.low_pass = low_pass
        image_object.file_directory = file_output_dir

        # Convert high-pass and low-pass to image format
        arr = high_pass
        arr = normalise_array(arr)
        img_dir = Path(file_output_dir) / "images"
        save_image(arr, img_dir, f"{filename}_high_pass.jpg", cmap="grey")

        arr = low_pass
        arr = normalise_array(arr)
        save_image(arr, img_dir, f"{filename}_low_pass.jpg", cmap="grey")
    else:
        logger.info(f"[{image_object.filename}] : Frequency splitting is disabled by config, the original image will be used.")
        if image_object.image_flattened is not None:
            image_object.high_pass = image_object.image_flattened
        else:
            image_object.high_pass = image_object.image_original

    arr = image_object.image_original
    arr = normalise_array(arr)
    save_image(arr, file_output_dir / "images", f"{filename}_original.jpg", cmap="grey")


def perform_fourier(
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

    freq_grid = create_frequency_mask(extended_image)

    # Create mask to filter to specified frequencies
    mask = apply_cutoff(freq_grid, cutoff, edge_width=edge_width)

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
        min_cutoff: float,
        max_cutoff: float,
        cutoff_step: float,
        min_rms: float,
        pixel_to_nm_scaling: float,
) -> float:
    """
    Find an ideal cutoff for performing a fourier transform on
    the current image.
    A frequency grid is created and then multiple cutoffs are tested
    between the cutoff bounds defined in the config. The cutoff with the
    lowest rms value over the minimum rms defined in config is then chosen
    for the rest of the process.

    Parameters
    ----------
    image_object : ImageData
        Class object of the current image containing all data
        connected to it.
    edge_width : float
        Edge width, expressed as a relative fraction of the Nyquist
        frequency.  If zero, the filter has sharp edges.  For non-zero
        values the transition has the shape of the error function,
        with the specified width.
    min_cutoff : float
        Starting cutoff when looping through to find the ideal value.
    max_cutoff : float
        Finishing cutoff when looping through to find the ideal value.
    cutoff_step : float
        Amount to increase tested cutoff by on each test.
    min_rms : float
        RMS threshold for when checking cutoffs.
    pixel_to_nm_scaling : float
        Ratio of pixels:nm used for scaling parameters to ensure
        consistency.

    Returns
    -------
    float
        The best cutoff found.
    """
    image = image_object.image_original
    best_cutoff = None
    min_found_rms = float('inf')

    extended_image, _ = extend_image(image, method=cv2.BORDER_REFLECT)
    fft_input = pyfftw.empty_aligned(extended_image.shape, dtype="complex128")
    fft_object = pyfftw.builders.fft2(fft_input)

    fft_input[:] = extended_image
    full_dft = fft_object()

    n_pixels = full_dft.size

    min_rms = min_rms + (pixel_to_nm_scaling * 0.1)

    freq_grid = create_frequency_mask(extended_image)

    for cutoff in np.arange(min_cutoff, max_cutoff, cutoff_step):
        mask = apply_cutoff(freq_grid, cutoff, edge_width=edge_width)
        masked_dft = full_dft * mask

        sum_sq_magnitudes = np.sum(np.abs(masked_dft)**2)
        current_rms = np.sqrt(sum_sq_magnitudes / (n_pixels**2))

        if current_rms > min_rms and current_rms < min_found_rms:
            min_found_rms = current_rms
            best_cutoff = cutoff

    if best_cutoff is None:
        logger.warning(
            f"[{image_object.filename}] : No cutoff could be found for the image. Please consider adjusting the cutoff bounds in the config."
        )
    return best_cutoff


def apply_cutoff(
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
    if edge_width and edge_width > 0:
        return 0.5 * (erf((f_grid - cutoff) / edge_width) + 1)
    else:
        return (f_grid >= cutoff).astype(np.float64)

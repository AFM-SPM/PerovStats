from __future__ import annotations
from pathlib import Path
import cv2

from loguru import logger
import numpy as np
from PIL import Image
import pyfftw
from matplotlib import pyplot as plt

from .core.classes import ImageData
from .core.image_processing import extend_image, calculate_rms, normalise_array
from .core.segmentation import (
    create_frequency_mask,
    create_grain_mask
)
from .smears import find_smear_areas



def split_frequencies(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    """
    Carry out frequency splitting on an image.

    Parameters
    ----------
    args : list[str], optional
        Arguments.

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
        img = Image.fromarray(arr * 255).convert("L")
        img_dir = Path(file_output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(file_output_dir / "images" / f"{filename}_high_pass.jpg")

        arr = low_pass
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_low_pass.jpg")
    else:
        logger.info(f"[{image_object.filename}] : Frequency splitting is disabled by config, the original image will be used.")
        if image_object.image_flattened is not None:
            image_object.high_pass = image_object.image_flattened
        else:
            image_object.high_pass = image_object.image_original

    arr = image_object.image_original
    arr = normalise_array(arr)
    img = Image.fromarray(arr * 255).convert("L")
    img.save(file_output_dir / "images" / f"{filename}_original.jpg")


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


def find_cutoff(
        image_object: ImageData,
        edge_width: float,
        min_cutoff: float,
        max_cutoff: float,
        cutoff_step: float,
        min_rms: float,
        pixel_to_nm_scaling: float,
    ) -> float:
    """Iterate through possible cutoff points to find one with the smallest RMS over a given value."""
    image = image_object.image_original
    best_cutoff = None
    min_found_rms = float('inf')
    cutoff_values = []
    rms_values = []
    # Adjust the minimum RMS based on the image scaling
    min_rms = min_rms + (pixel_to_nm_scaling * 0.1)

    for cutoff in np.arange(min_cutoff, max_cutoff, cutoff_step):
        high_pass, _ = perform_fourier(image, cutoff, edge_width)
        current_rms = calculate_rms(high_pass)

        cutoff_values.append(cutoff)
        rms_values.append(current_rms)

        if current_rms > min_rms and current_rms < min_found_rms:
            min_found_rms = current_rms
            best_cutoff = cutoff

    if best_cutoff is None:
        logger.warning(
            f"[{image_object.filename}] : No cutoff could be found for the image. Skipping.."
        )
    return best_cutoff


def run_frequency_splitting(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    split_frequencies(config, image_object)

    output_dir = Path(config["output_dir"])

    if image_object.high_pass is not None:
        # For each image create and save a mask
        fname = image_object.filename
        im = image_object.high_pass
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

        # Remove/ ignore smears in high_pass image
        smear_config = config["remove_smears"]
        if smear_config["run"]:
            image_object.smears, smears_removed = find_smear_areas(image_object.high_pass, image_object.low_pass, smear_config, fname)
            image_object.smears_removed = smears_removed

            rgb_highpass = np.stack((image_object.high_pass,)*3, axis=-1)
            rgb_highpass = normalise_array(rgb_highpass)
            rgb_highpass[image_object.smears > 0] = [1, 0, 0]

        # Scale threshold block size with image scaling and round to nearest odd integer
        threshold_block_size = config["segmentation"]["threshold_block_size"] / pixel_to_nm_scaling
        threshold_block_size = 2 * round((threshold_block_size - 1) / 2) + 1

        # Cleaning config options - adjusted for pixel to nm scaling
        area_threshold = config["segmentation"]["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = config["segmentation"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options - adjusted for pixel to nm scaling
        smooth_sigma = config["segmentation"]["smoothing"]["sigma"]
        if smooth_sigma:
            smooth_sigma = smooth_sigma / pixel_to_nm_scaling

        logger.info(f"[{image_object.filename}] : *** Mask creation ***")
        logger.info(f"[{image_object.filename}] : Creating grain mask")
        np_mask = create_grain_mask(
            im,
            threshold_block_size=threshold_block_size,
            smooth_sigma=smooth_sigma,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
        )

        image_object.mask = np_mask

        # Convert to image format and save
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask.jpg", np_mask)

        # Save high-pass with mask skeleton overlay
        high_pass = image_object.high_pass
        rgb_highpass = np.stack((high_pass,)*3, axis=-1)
        rgb_highpass = normalise_array(rgb_highpass)
        rgb_highpass[np_mask > 0] = [1, 0, 0]
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask_overlay.jpg", rgb_highpass)

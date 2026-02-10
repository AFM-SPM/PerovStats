from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skimage as ski
from skimage.measure import label

from perovstats.grains import tidy_border, label2rgb, regionprops
from .freqsplit import frequency_split, find_cutoff
from .segmentation import create_grain_mask
from .segmentation import threshold_mad, threshold_mean_std


def create_masks(config, image_object) -> None:
    split_frequencies(config, image_object)

    output_dir = Path(config["output_dir"])

    if image_object.high_pass is not None:
        # For each image create and save a mask
        fname = image_object.filename
        im = image_object.high_pass
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

        # Thresholding config options
        threshold_func = config["mask"]["threshold_function"]
        if threshold_func == "mad":
            threshold_func = threshold_mad
        elif threshold_func == "std":
            threshold_func = threshold_mean_std
        min_threshold = config["mask"]["threshold_bounds"][0]
        max_threshold = config["mask"]["threshold_bounds"][1]

        # Cleaning config options
        area_threshold = config["mask"]["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = config["mask"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options
        smooth_sigma = config["mask"]["smoothing"]["sigma"]
        smooth_func = config["mask"]["smoothing"]["smooth_function"]
        if smooth_func == "gaussian":
            smooth_func = ski.filters.gaussian
        elif smooth_func == "difference_of_gaussians":
            smooth_func = ski.filters.difference_of_gaussians

        logger.info(f"[{image_object.filename}] : *** Mask creation ***")

        threshold = find_threshold(
            image_object.filename,
            im,
            threshold_func=threshold_func,
            smooth_sigma=smooth_sigma,
            smooth_func=smooth_func,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        if threshold is None:
            return

        image_object.threshold = threshold

        logger.info(f"[{image_object.filename}] : Creating grain mask")
        np_mask = create_grain_mask(
            im,
            threshold_func=threshold_func,
            threshold=threshold,
            smooth_sigma=smooth_sigma,
            smooth_func=smooth_func,
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


def split_frequencies(config, image_object) -> list[np.real]:
    """
    Carry out frequency splitting on a batch of files.

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
        )

        if not cutoff:
            return

        cutoff_nm = 2 * pixel_to_nm_scaling / cutoff
        logger.info(f"[{image_object.filename}] : Frequency cutoff: {cutoff} ({np.round(cutoff_nm, 4)}nm)")

        # Update image class with chosen cutoff
        image_object.cutoff = cutoff
        image_object.cutoff_freq_nm = cutoff_nm

        logger.info(f"[{filename}] : Splitting image frequencies")
        high_pass, low_pass = frequency_split(
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


def normalise_array(arr):
    v_min, v_max = np.percentile(arr, [0.05, 99.95])

    clipped = np.clip(arr, v_min, v_max)
    normalised = (clipped - v_min) / (v_max - v_min)
    return normalised


def find_threshold(
    filename: str,
    image: np.ndarray,
    threshold_func: callable,
    smooth_sigma: float,
    smooth_func,
    area_threshold,
    disk_radius,
    pixel_to_nm_scaling,
    min_threshold,
    max_threshold,
):
    """
    Loop through possible threshold values and select the value
    that produces the most grains.

    Parameters
    ----------
    filename: str
        Name of the image being processed.
    image : np.ndarray
        Numpy array of the high-passed image to use.
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
        mask_areas = [
            regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
        ]

        if len(mask_areas) >= best_grain_num:
            best_grain_num = len(mask_areas)
            best_threshold = curr_threshold

    if best_grain_num == 0:
        logger.warning(f"[{filename}] : No grains could be found for any tested threshold.",
                       "consider increasing the threshold bounds in the config.",
                       "Skipping image..")
        return None

    logger.info(f"[{filename}] : Best threshold found: {best_threshold}")
    return best_threshold

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skimage as ski
from skimage.measure import label

from perovstats.grains import tidy_border, label2rgb, regionprops
from .freqsplit import frequency_split, find_cutoff
from .segmentation import create_grain_mask
from .segmentation import threshold_mad, threshold_mean_std


LOGGER = logging.getLogger(__name__)


def create_masks(perovstats_object) -> None:
    split_frequencies(perovstats_object)

    output_dir = Path(perovstats_object.config["output_dir"])

    for i, image in enumerate(perovstats_object.images):
        # For each image create and save a mask
        fname = image.filename
        im = image.high_pass
        pixel_to_nm_scaling = image.pixel_to_nm_scaling

        # Thresholding config options
        threshold_func = perovstats_object.config["mask"]["threshold_function"]
        if threshold_func == "mad":
            threshold_func = threshold_mad
        elif threshold_func == "std":
            threshold_func = threshold_mean_std

        # Cleaning config options
        area_threshold = perovstats_object.config["mask"]["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = perovstats_object.config["mask"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options
        smooth_sigma = perovstats_object.config["mask"]["smoothing"]["sigma"]
        smooth_function = perovstats_object.config["mask"]["smoothing"]["smooth_function"]
        if smooth_function == "gaussian":
            smooth_function = ski.filters.gaussian
        elif smooth_function == "difference_of_gaussians":
            smooth_function = ski.filters.difference_of_gaussians

        threshold = find_threshold(
            im,
            threshold_func=threshold_func,
            smooth_sigma=smooth_sigma,
            smooth_function=smooth_function,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
        )

        np_mask = create_grain_mask(
            im,
            threshold_func=threshold_func,
            threshold=threshold,
            smooth_sigma=smooth_sigma,
            smooth_function=smooth_function,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
        )

        perovstats_object.images[i].mask = np_mask

        # Convert to image format and save
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask.jpg", np_mask)

        # Save high-pass with mask skeleton overlay
        high_pass = perovstats_object.images[i].high_pass
        rgb_highpass = np.stack((high_pass,)*3, axis=-1)
        rgb_highpass[np_mask > 0] = [1, 0, 0]
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask_overlay.jpg", rgb_highpass)


def split_frequencies(perovstats_object) -> list[np.real]:
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
    cutoff_freq_nm = perovstats_object.config["freqsplit"]["cutoff_freq_nm"]
    edge_width = perovstats_object.config["freqsplit"]["edge_width"]
    output_dir = Path(perovstats_object.config["output_dir"])

    for image_data in perovstats_object.images:
        filename = image_data.filename

        file_output_dir = Path(output_dir / filename)
        file_output_dir.mkdir(parents=True, exist_ok=True)

        if image_data.image_flattened is not None:
            image = image_data.image_flattened
        else:
            image = image_data.image_original
        pixel_to_nm_scaling = image_data.pixel_to_nm_scaling
        LOGGER.debug("[%s] Image dimensions: ", image.shape)
        LOGGER.info("[%s] : *** Frequency splitting ***", filename)

        if cutoff_freq_nm:
            cutoff = 2 * pixel_to_nm_scaling / cutoff_freq_nm

        LOGGER.info("[%s] : pixel_to_nm_scaling: %s", filename, pixel_to_nm_scaling)

        cutoff, rmses, cutoffs = find_cutoff(
            image,
            edge_width,
            min_cutoff=0,
            max_cutoff=0.2,
            cutoff_step=0.005,
            min_rms=12,
        )

        cutoff_nm = 2 * pixel_to_nm_scaling / cutoff
        print(f"CHOSEN CUTOFF: {cutoff} ({round(cutoff_nm, 3)}nm)")

        # Update image class with chosen cutoff
        image_data.cutoff = cutoff
        image_data.cutoff_freq_nm = cutoff_nm

        high_pass, low_pass = frequency_split(
            image,
            cutoff=cutoff,
            edge_width=edge_width,
        )

        image_data.high_pass = normalise_array(high_pass)
        image_data.low_pass = low_pass
        image_data.file_directory = file_output_dir

        # Convert to image format
        arr = high_pass
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img_dir = Path(file_output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(file_output_dir / "images" / f"{filename}_high_pass.jpg")

        arr = low_pass
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_low_pass.jpg")

        arr = image_data.image_original
        # arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = normalise_array(arr)
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_original.jpg")


def normalise_array(arr):
    v_min, v_max = np.percentile(arr, [0.05, 99.95])

    clipped = np.clip(arr, v_min, v_max)
    normalised = (clipped - v_min) / (v_max - v_min)
    return normalised


def find_threshold(
        image,
        threshold_func,
        smooth_sigma,
        smooth_function,
        area_threshold,
        disk_radius,
        pixel_to_nm_scaling,
    ):
    """
    Loop through possible threshold values and select the value
    that produces the most grains.

    Parameters
    ----------
    image : np.ndarray
        Numpy array of the high-passed image to use.
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
    pixel_to_nm_scaling : float
        The scale factor of pixels:nm

    Returns
    -------
    float
        The selected best threshold.
    """
    best_threshold = None
    best_grain_num = 0
    for curr_threshold in np.arange(-1, 3, 0.01):
        curr_threshold = round(curr_threshold, 3)
        np_mask = create_grain_mask(
            image,
            threshold_func=threshold_func,
            threshold=curr_threshold,
            smooth_sigma=smooth_sigma,
            smooth_function=smooth_function,
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

        if len(mask_areas) > best_grain_num:
            best_grain_num = len(mask_areas)
            best_threshold = curr_threshold

    print(f"BEST THRESHOLD FOUND: {best_threshold} which finds {best_grain_num} grains.")

    return best_threshold, best_grain_num

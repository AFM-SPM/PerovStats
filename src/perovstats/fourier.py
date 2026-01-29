from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import skimage as ski
from skimage.filters import threshold_local
from matplotlib import pyplot as plt

from .freqsplit import frequency_split, frequency_split_2
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
        threshold = perovstats_object.config["mask"]["threshold"]
        threshold_func = perovstats_object.config["mask"]["threshold_function"]
        if threshold_func == "mad":
            threshold_func = threshold_mad
        elif threshold_func == "std":
            threshold_func = threshold_mean_std

        # Cleaning config options
        area_threshold = perovstats_object.config["mask"]["cleaning"]["area_threshold"] / (pixel_to_nm_scaling**2)
        disk_radius = perovstats_object.config["mask"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling

        # Smoothing config options
        smooth_sigma = perovstats_object.config["mask"]["smoothing"]["sigma"]
        smooth_function = perovstats_object.config["mask"]["smoothing"]["smooth_function"]
        if smooth_function == "gaussian":
            smooth_function = ski.filters.gaussian

        np_mask = create_grain_mask(
            im,
            threshold_func=threshold_func,
            threshold=threshold,
            smooth_sigma=smooth_sigma,
            smooth_function=smooth_function,
            area_threshold=area_threshold,
            disk_radius=disk_radius
        )

        perovstats_object.images[i].mask = np_mask

        # Convert to image format and save
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask.jpg", np_mask)


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
        LOGGER.info("[%s] : cutoff: %s, edge_width: %s", filename, cutoff, edge_width)

        high_pass, low_pass = frequency_split(
            image,
            cutoff=cutoff,
            edge_width=edge_width,
            pixel_to_nm_scaling=pixel_to_nm_scaling,
        )

        image_data.high_pass = high_pass
        image_data.low_pass = low_pass
        image_data.file_directory = file_output_dir

        # Convert to image format
        arr = high_pass
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        img = Image.fromarray(arr * 255).convert("L")
        img_dir = Path(file_output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(file_output_dir / "images" / f"{filename}_high_pass.jpg")

        arr = low_pass
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_low_pass.jpg")

        arr = image_data.image_original
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / "images" / f"{filename}_original.jpg")

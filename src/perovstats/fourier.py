from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import skimage as ski
from matplotlib import pyplot as plt
from yaml import dump as dump_yaml
from yaml import safe_load
from yaml import safe_dump

from .freqsplit import frequency_split
from .segmentation import clean_mask
from .segmentation import create_grain_mask
from .segmentation import threshold_mad
from .classes import Mask


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path("./images")
OUTPUT_DIR = Path("./output")
CUTOFF_FREQ_NM = ["225", "250", "275", "300", "325"]

# Dataset configuration
DATA_CONFIG = [
    {
        "data_dir": "1st samples- just retrace - 1-2um pyramids/C60 data",
        "height_channel": "Height Sensor",
        "file_ext": ".spm",
    },
    {
        "data_dir": "1st samples- just retrace - 1-2um pyramids/PFQNM",
        "height_channel": "Height Sensor",
        "file_ext": ".spm",
    },
    {
        "data_dir": "1st samples- just retrace - 1-2um pyramids/Tapping mode",
        "height_channel": "Height Sensor",
        "file_ext": ".spm",
    },
    {
        "data_dir": "2nd sample batch - just retrace - 1-2um pyramids",
        "height_channel": "Height Sensor",
        "file_ext": ".spm",
    },
    {
        "data_dir": "2nd sample batch - trace and retrace - 1-2um pyramids",
        "height_channel": "Height",
        "file_ext": ".spm",
    },
    {
        "data_dir": "3rd sample batch -Different sized pyramids/1-2um pyramids - PFQNM",
        "height_channel": "Height",
        "file_ext": ".spm",
    },
    {
        "data_dir": "4th batch - trace and retrace - rehmat - MFP3d microscope - feb 2025/25_02_06_ST4-14-31",
        "height_channel": "HeightTrace",
        "file_ext": ".ibw",
    },
    {
        "data_dir": "4th batch - trace and retrace - rehmat - MFP3d microscope - feb 2025/25_02_10_ST4-14-31",
        "height_channel": "HeightTrace",
        "file_ext": ".ibw",
    },
    {
        "data_dir": "NuNano Scout AFM Tip - trace and retrace - 1-2um pyramids",
        "height_channel": "Height",
        "file_ext": ".spm",
    },
]

THRESHOLD_FUN = threshold_mad
THRESHOLD_ARGS = {"k": 12}
SMOOTH_FUN = ski.filters.gaussian
DIFF_GAUSS_SIGMA = (1, 3)
GAUSS_SIGMA = 8
CLEAN_FUN = clean_mask
AREA_THRESHOLD_NM2 = 10000
DISK_RADIUS_FACTOR = 40

if ski.filters.gaussian == SMOOTH_FUN:
    SMOOTH_ARGS = {"sigma": GAUSS_SIGMA}
elif ski.filters.difference_of_gaussians == SMOOTH_FUN:
    SMOOTH_ARGS = dict(zip(["low_sigma", "high_sigma"], DIFF_GAUSS_SIGMA))
else:
    SMOOTH_ARGS = {}

# Segmentation mask configuration
MASK_CONFIG = {
    "threshold": THRESHOLD_FUN,
    "threshold_args": THRESHOLD_ARGS,
    "smooth": SMOOTH_FUN,
    "smooth_args": SMOOTH_ARGS,
    "clean": CLEAN_FUN,
}

def plot_compare(
    im1: np.ndarray,
    im2: np.ndarray,
    figsize: tuple[float] = (10, 8),
    overlay: bool = False,
    title: str | None = None,
) -> None:
    """Compare two plots side by side."""
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    ax[0].imshow(im1)
    ax[0].set_title("Original")
    if overlay:
        ax[1].imshow(im1, cmap="gray")
        ax[1].imshow(im2, cmap="jet", alpha=0.25)
    else:
        ax[1].imshow(im2)
        ax[1].set_title("Mask")
    if title:
        fig.suptitle(title)
    plt.show()


def create_masks(image_dicts, fs_config) -> None:
    # for cutoff_freq_nm in CUTOFF_FREQ_NM:
    #     for config in DATA_CONFIG:
    #         output_dir = Path(
    #             OUTPUT_DIR,
    #             f"cutoff_freq_nm-{cutoff_freq_nm}",
    #             config["data_dir"],
    #         )
    #         arg_list = [
    #             "-d",
    #             Path(BASE_DIR, config["data_dir"]).as_posix(),
    #             "-e",
    #             config["file_ext"],
    #             "-n",
    #             config["height_channel"],
    #             "-f",
    #             cutoff_freq_nm,
    #             "-o",
    #             output_dir.as_posix(),
    #         ]
    split_frequencies(image_dicts, fs_config)

    output_dir = Path(fs_config["output_dir"])
    files = sorted(output_dir.glob("*\*_high_pass.npy"))

    masks = []
    for f in files:
        fname = f.name.replace("_high_pass.npy", "")
        im = np.load(f)
        with (f.parent / f"{fname}_config.yaml").open() as conf:
            fs_config = safe_load(conf)
        pixel_to_nm_scaling = fs_config["pixel_to_nm_scaling"]
        mask_config = MASK_CONFIG.copy()
        mask_config["clean_args"] = {
            "area_threshold": AREA_THRESHOLD_NM2 / (pixel_to_nm_scaling**2),
            "disk_radius": DISK_RADIUS_FACTOR / pixel_to_nm_scaling,
        }
        mask = create_grain_mask(im, **mask_config)
        np.save(output_dir / fname / f"{fname}_mask.npy", mask)
        with Path(output_dir / fname / f"{fname}_mask.yaml").open("w") as outfile:
            dump_yaml(mask_config, outfile, default_flow_style=False)
        # Convert to image format
        plt.imsave(output_dir / fname / f"{fname}_mask.jpg", mask)
        # View images
        # plot_compare(im, mask, figsize=(15, 6), overlay=True, title=fname)

        fdir = output_dir / fname
        masks.append(Mask(mask, fname, fdir, fs_config))

    return masks


def split_frequencies(image_dicts, fs_config) -> list[np.real]:
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
    cutoff_freq_nm = fs_config["cutoff_freq_nm"]
    edge_width = fs_config["edge_width"]
    output_dir = Path(fs_config["output_dir"])

    for filename, topostats_object in image_dicts.items():
        fs_config["filename"] = filename

        file_output_dir = Path(output_dir / filename)
        file_output_dir.mkdir(parents=True, exist_ok=True)
        # Save original image data
        # np.save(
        #     file_output_dir / f"{filename}_original.npy",
        #     topostats_object["image_original"],
        # )

        # LOGGER.info("[%s] : Saved original to %s_original.npy", filename, filename)

        if topostats_object.get("image_flattened") is not None:
            image = topostats_object["image_flattened"]
        else:
            image = topostats_object["image_original"]
        pixel_to_nm_scaling = topostats_object["pixel_to_nm_scaling"]
        fs_config["pixel_to_nm_scaling"] = pixel_to_nm_scaling
        LOGGER.debug("[%s] Image dimensions: ", image.shape)
        LOGGER.info("[%s] : *** Frequency splitting ***", filename)

        if cutoff_freq_nm:
            cutoff = 2 * pixel_to_nm_scaling / cutoff_freq_nm
            fs_config["cutoff"] = cutoff

        LOGGER.info("[%s] : pixel_to_nm_scaling: %s", filename, pixel_to_nm_scaling)
        LOGGER.info("[%s] : cutoff: %s, edge_width: %s", filename, cutoff, edge_width)

        high_pass, low_pass = frequency_split(
            image,
            cutoff=cutoff,
            edge_width=edge_width,
        )

        # Save high pass image data
        np.save(file_output_dir / f"{filename}_high_pass.npy", high_pass)

        # Save low pass image data
        # np.save(output_dir / f"{filename}_low_pass.npy", low_pass)

        LOGGER.info("[%s] : Saved output to %s_high_pass.npy", filename, filename)

        # Convert to image format
        arr = high_pass
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        img = Image.fromarray(arr * 255).convert("L")
        img.save(file_output_dir / f"{filename}_high_pass.jpg")

        #arr = low_pass
        #arr = (arr - arr.min()) / (arr.max() - arr.min())
        #img = Image.fromarray(arr * 255).convert("L")
        #img.save(file_output_dir / f"{filename}_low_pass.jpg")

        #arr = topostats_object["image_original"]
        #arr = (arr - arr.min()) / (arr.max() - arr.min())
        #img = Image.fromarray(arr * 255).convert("L")
        #img.save(file_output_dir / f"{filename}_original.jpg")

        # Save configuration metadata for frequency splitting
        with Path(file_output_dir / f"{filename}_config.yaml").open("w") as outfile:
            safe_dump(fs_config, outfile, default_flow_style=False)

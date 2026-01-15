from __future__ import annotations
from pathlib import Path
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from yaml import dump as dump_yaml
from yaml import safe_load
from perovstats.cli import freqsplit
from perovstats.segmentation import clean_mask
from perovstats.segmentation import create_grain_mask
from perovstats.segmentation import threshold_mad

# Data directory
BASE_DIR = Path("../images chosen")

# Output directory
OUTPUT_DIR = Path("./output")

# Frequency split cutoff in nm
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
THRESHOLD_ARGS = {"k": -0.8}
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


for cutoff_freq_nm in CUTOFF_FREQ_NM:
    for config in DATA_CONFIG:
        output_dir = Path(
            OUTPUT_DIR,
            f"cutoff_freq_nm-{cutoff_freq_nm}",
            config["data_dir"],
        )
        arg_list = [
            "-d",
            Path(BASE_DIR, config["data_dir"]).as_posix(),
            "-e",
            config["file_ext"],
            "-n",
            config["height_channel"],
            "-f",
            cutoff_freq_nm,
            "-o",
            output_dir.as_posix(),
        ]
        freqsplit(arg_list)

        files = sorted(output_dir.glob("*_high_pass.npy"))
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
            np.save(output_dir / f"{fname}_mask.npy", mask)
            with Path(output_dir / f"{fname}_mask.yaml").open("w") as outfile:
                dump_yaml(mask_config, outfile, default_flow_style=False)
            # Convert to image format
            plt.imsave(output_dir / f"{fname}_mask.jpg", mask)
            # View images
            plot_compare(im, mask, figsize=(15, 6), overlay=True, title=fname)

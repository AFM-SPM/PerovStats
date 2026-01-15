from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from loguru import logger
from ruamel.yaml import YAML
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import regionprops

# Data directory
DATA_DIR = Path("../output")

# Annotated data
DATA_ANNOTATED = Path("path/to/perovskites_data_notated.csv")

NM_TO_MICRON = 1e-3


config_yaml_files = list(DATA_DIR.glob("*/**/*_config.yaml"))
logger.info(f"found {len(config_yaml_files)} config files")
print(config_yaml_files)


files_to_include = []
if DATA_ANNOTATED.exists():
    data_annotated = pd.read_csv(DATA_ANNOTATED)
    data_to_include = data_annotated[data_annotated.include == "Y"]
    files_to_include = data_to_include.filename.apply(Path, axis=1)
names = [f.stem for f in files_to_include]


def plot_areas(areas: list, title: str | None = None, units: str = "um") -> None:
    """Plot histogram of mask areas."""
    if title is None:
        title = ""
    title = title + f" n:{len(areas)}"
    plt.figure()
    if units == "um":
        areas = [area * NM_TO_MICRON**2 for area in areas]
        plt.xlabel("area (µm²)")
    elif units == "nm":
        plt.xlabel("area (nm²)")
    else:
        msg = "units must be 'um' or 'nm'"
        raise ValueError(msg)
    sns.histplot(areas, kde=True, bins="auto", log_scale=True)
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_mask_gallery(
    masks_data: dict[str, dict[str, npt.NDArray | float]],
    col_num: int = 3,
) -> None:
    """Plot mask gallery."""
    num_rows = len(masks_data) // col_num + 1

    fig, ax = plt.subplots(num_rows, col_num, figsize=(8 * col_num, 8 * num_rows))
    for i, (filename, mask_data) in enumerate(masks_data.items()):
        mask_rgb = mask_data["mask_rgb"]
        num_grains = mask_data["num_grains"]
        grains_per_nm2 = mask_data["grains_per_nm2"]
        grains_per_um2 = grains_per_nm2 / NM_TO_MICRON**2
        mask_size_x_um = mask_data["mask_size_x_nm"] * NM_TO_MICRON
        mask_size_y_um = mask_data["mask_size_y_nm"] * NM_TO_MICRON
        title = (
            f"{filename}\n"
            f"image size: {mask_size_x_um} x {mask_size_y_um} µm² | "
            f"grains: {num_grains} | grains/µm²: {grains_per_um2:.2f}"
        )
        row = i // col_num
        col = i % col_num
        ax[row, col].imshow(mask_rgb, cmap="gray")
        ax[row, col].set_title(title)
    plt.tight_layout()
    plt.show()


all_masks_grain_areas = []
all_masks_data = {}
data = []

for config_yaml_file in config_yaml_files:
    filename = config_yaml_file.name[: -len("_config.yaml")]

    # Only analyse selected list of files, if specified
    if names and filename not in names:
        continue

    file_directory = config_yaml_file.parent

    logger.info(f"processing file {filename:<50} in '{file_directory}'")

    yaml = YAML()
    with config_yaml_file.open("r") as f:
        config_yaml = yaml.load(f)

    pixel_to_nm_scaling = config_yaml["pixel_to_nm_scaling"]

    mask_file = file_directory / f"{filename}_mask.npy"
    mask = np.load(mask_file).astype(bool)
    mask = np.invert(mask)
    labelled_mask = label(mask, connectivity=1)
    labelled_mask_rgb = label2rgb(labelled_mask, bg_label=0)

    mask_regionprops = regionprops(labelled_mask)
    mask_areas = [
        regionprop.area * pixel_to_nm_scaling**2 for regionprop in mask_regionprops
    ]
    all_masks_grain_areas.extend(mask_areas)

    mask_size_x_nm = mask.shape[1] * pixel_to_nm_scaling
    mask_size_y_nm = mask.shape[0] * pixel_to_nm_scaling
    mask_area_nm = mask_size_x_nm * mask_size_y_nm
    grains_per_nm2 = len(mask_areas) / mask_area_nm

    mask_data = {
        "mask_rgb": labelled_mask_rgb,
        "grains_per_nm2": grains_per_nm2,
        "mask_size_x_nm": mask_size_x_nm,
        "mask_size_y_nm": mask_size_y_nm,
        "mask_area_nm": mask_area_nm,
        "num_grains": len(mask_areas),
    }
    all_masks_data[f"{filename}-{config_yaml['cutoff_freq_nm']}"] = mask_data

    data.append(
        {
            "filename": filename,
            "grains_per_nm2": grains_per_nm2,
            "mask_size_x_nm": mask_size_x_nm,
            "mask_size_y_nm": mask_size_y_nm,
            "mask_area_nm": mask_area_nm,
            "num_grains": len(mask_areas),
            "dir": config_yaml_file.parent,
            "cutoff_freq_nm": config_yaml["cutoff_freq_nm"],
            "cutoff": config_yaml["cutoff"],
        },
    )

grain_stats = pd.DataFrame(data)

logger.info(
    f"obtained {grain_stats['num_grains'].sum()} grains in {len(grain_stats)} masks",
)

plot_areas(all_masks_grain_areas, title="all masks areas", units="nm")

plot_mask_gallery(all_masks_data, col_num=3)


sns.histplot(
    grain_stats,
    x="grains_per_nm2",
    hue="mask_size_x_nm",
    kde=True,
    bins="auto",
    log_scale=True,
)
plt.show()

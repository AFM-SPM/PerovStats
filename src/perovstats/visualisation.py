from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import skimage as ski
from loguru import logger
import numpy as np
import numpy.typing as npt
import seaborn as sns

CUTOFF_FREQ_NM = 250

# Whether to show original side-by-side with mask overlay
SHOW_ORIG = False

# Whether to show dynamic plots
PLOT_DYNAMIC = True

# Display size for dynamic plots
DISPLAY_SIZE = 850

# Whether to enable annotations for dynamic plots
ENABLE_ANNOTATIONS = True

# Path to frequency split samples
DATA_DIR = Path(
    f"./freqsplit_samples_20250424/cutoff_freq_nm-{CUTOFF_FREQ_NM}/",
)

# Path to annotated data
DATA_ANNOTATED = Path(
    "./perovskites_data_notated.csv",
)


def create_plots(
        filename: str,
        all_masks_grain_areas: list,
        all_masks_data: dict[str, dict[str, npt.NDArray | float]],
        nm_to_micron: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title("Grain size distributions")
    plot_areas(all_masks_grain_areas, nm_to_micron, title="all masks areas", units="nm", ax=axes[0])
    for i, (filename, mask_data) in enumerate(all_masks_data.items()):
        plot_coloured_grains(filename, nm_to_micron, mask_data, col_num=1, ax=axes[i + 1])
    plt.tight_layout()
    plt.show()


def plot_areas(areas: list, nm_to_micron: float, title: str | None = None, units: str = "um", ax=None) -> None:
    """Plot histogram of mask areas."""
    if ax is None:
        plt.gca()
    if title is None:
        title = ""
    title = title + f" n:{len(areas)}"
    if units == "um":
        areas = [area * nm_to_micron**2 for area in areas]
        ax.set_xlabel("area (µm²)")
    elif units == "nm":
        ax.set_xlabel("area (nm²)")
    else:
        msg = "units must be 'um' or 'nm'"
        raise ValueError(msg)
    sns.histplot(areas, kde=True, bins="auto", log_scale=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("count")


def plot_coloured_grains(
        filename: str,
        nm_to_micron: float,
        mask_data: dict[str, dict[str, npt.NDArray | float]],
        col_num: int = 1,
        ax = None,
) -> None:
    """Plot coloured grains."""
    if ax is None:
        ax = plt.gca()
    # num_items = len(masks_data)
    # num_rows = (num_items + col_num - 1) // col_num
    # fig, ax = plt.subplots(num_rows, col_num, figsize=(6 * col_num, 6 * num_rows))
    # for i, (filename, mask_data) in enumerate(masks_data.items()):
    mask_rgb = mask_data["mask_rgb"]
    num_grains = mask_data["num_grains"]
    grains_per_nm2 = mask_data["grains_per_nm2"]
    grains_per_um2 = grains_per_nm2 / nm_to_micron**2
    mask_size_x_um = mask_data["mask_size_x_nm"] * nm_to_micron
    mask_size_y_um = mask_data["mask_size_y_nm"] * nm_to_micron
    title = (
        f"{filename}\n"
        f"image size: {mask_size_x_um} x {mask_size_y_um} µm² | "
        f"grains: {num_grains} | grains/µm²: {grains_per_um2:.2f}"
    )
    # row = i // col_num
    # col = i % col_num
    # ax = np.atleast_2d(ax)
    # ax[row, col].imshow(mask_rgb, cmap="gray")
    # ax[row, col].set_title(title)
    ax.imshow(mask_rgb)
    ax.set_title(title)
    ax.axis("off")


def plot_compare(
    ims: list[np.ndarray],
    titles: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Plot images side-by-side for comparison."""
    n = len(ims)
    fig, ax = plt.subplots(ncols=n, figsize=(5 * n, 8))
    for i in range(n):
        ax[i].imshow(ims[i])
        if titles:
            ax[i].set_title(titles[i])
    if title:
        fig.suptitle(title, y=0.8)
    # plt.show()


def plot_overlay(im: np.ndarray, mask: np.ndarray, show_orig: bool = False) -> None:
    """Plot image and mask overlay."""
    n = 2 if show_orig else 1
    fig, axs = plt.subplots(ncols=n, figsize=(12, 12))
    if show_orig:
        axs[0].imshow(im)
        ax = axs[1]
    else:
        ax = axs
    ax.imshow(im, cmap="gray")
    ax.imshow(mask > 0, cmap="jet", alpha=0.2)
    # plt.show()


def plot_dynamic(
    ims: list[np.ndarray],
    mask: np.ndarray,
    apply_mask: int = -1,
    title: str = "",
    display_size: int = 800,
    enable_annotations: bool = True,
) -> None:
    """Plot images with dynamic mask overlay.

    Requires plotly >= 4.14.
    """
    imshow_args = {
        "animation_frame": 0,
        "binary_string": False,
        "labels": {"animation_frame": "image"},
        "width": display_size,
        "height": display_size,
    }
    ims_to_show = [ski.exposure.rescale_intensity(im, out_range=(0, 1)) for im in ims]
    # create mask overlay image for display
    overlay = ims_to_show[apply_mask].copy()
    overlay[mask] = 1
    img = np.stack([*ims_to_show, overlay])
    fig = px.imshow(img, **imshow_args)
    fig.update_layout(coloraxis_showscale=False)
    if enable_annotations:
        # Set newshape properties; add modebar buttons; dragmode not set to leave as default (zoom)
        fig.update_layout(
            newshape={"line_color": "cyan"},
            title_text=f"{title}<br><em>Drag to add annotations - use modebar to change drawing tool</em>",
            margin_t=display_size * 0.15,
            font={"size": 10},
        )
        fig.show(
            config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ],
            },
        )
    else:
        fig.update_layout(title_text=title)
        fig.show()


def find_files() -> list[str]:
    files_to_include = []
    if DATA_ANNOTATED.exists():
        data_annotated = pd.read_csv(DATA_ANNOTATED)
        data_to_include = data_annotated[data_annotated.include == "Y"]
        files_to_include = data_to_include[["parent", "filename"]].apply(
            lambda row: Path(*row),
            axis=1,
        )
    else:
        # Create list of all files
        config_yaml_files = list(DATA_DIR.glob("**/*_config.yaml"))
        files_to_include = [
            p.with_name(p.name[: -len("_config.yaml")]) for p in config_yaml_files
        ]
    logger.info(f"Found {len(files_to_include)} files to include")
    return files_to_include


def plot_data(files_to_include: list[str]) -> None:
    for p in files_to_include:
        file_path = Path(DATA_DIR, p).with_suffix("")
        high_pass, mask, orig = map(
            np.load,
            sorted(file_path.parent.glob(f"{file_path.name}*.npy")),
        )
        titles = ["Original", f"Frequency split ({CUTOFF_FREQ_NM} nm)", "Mask"]
        plot_compare(
            [orig, high_pass, mask],
            titles=titles,
            title=file_path.relative_to(DATA_DIR),
        )
        plot_overlay(high_pass, mask, show_orig=SHOW_ORIG)
        if PLOT_DYNAMIC:
            plot_dynamic(
                [orig, high_pass],
                mask,
                title=file_path.relative_to(DATA_DIR),
                display_size=DISPLAY_SIZE,
                enable_annotations=ENABLE_ANNOTATIONS,
            )

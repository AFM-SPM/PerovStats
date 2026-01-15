from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import skimage as ski
from loguru import logger

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
    plt.show()


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
    plt.show()


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

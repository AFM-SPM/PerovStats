from __future__ import annotations
from pathlib import Path

from yaml import safe_dump
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from skimage import morphology
from loguru import logger
from topostats.plottingfuncs import Images

from .classes import ImageData
from .image_processing import normalise_array


def save_images(config: dict[str, any], image_object: ImageData, variation: str=None) -> None:
    """
    Use the image_set config list to save all requested plots and images.

    Parameters
    ----------
    config : dict[str, any]
        Dictionary of configuration options for the run.
    image_object : ImageData
        Dataclass instance containing all data for the image being processed.
    variation : str
        For notebooks processing the same data for each segmentation method.
    """
    cmap = config["colour_scheme"]
    get_cmap = cm.get_cmap(cmap)
    mask_cmap = 'bwr'
    output_dir = Path(config["output_dir"])
    filename = image_object.filename
    file_output_dir = Path(output_dir / filename)
    file_output_dir.mkdir(parents=True, exist_ok=True)
    if variation:
        save_dir = Path(config["output_dir"]) / filename / "images" / variation
    else:
        save_dir = Path(config["output_dir"]) / filename / "images"

    new_mask = image_object.mask.copy()
    new_mask[image_object.edge_grains] = 0
    if config["remove_smears"]["run"]:
        new_mask[image_object.smear_grains] = 0
    # Remove single pixels left in the smear area by accident
    new_mask = morphology.remove_small_objects(new_mask, max_size=1, connectivity=2)
    image_object.cleaned_mask = new_mask

    image_set = config["output"]["image_set"]

    if "highpass_mask" in image_set:
        # Save high-pass with mask overlay
        high_pass = image_object.high_pass
        vmin = high_pass.min()
        vmax = high_pass.max()
        Images(
            data=high_pass,
            output_dir=save_dir,
            filename=f"{filename}_highpass_mask_overlay",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highpass with mask",
            cmap=cmap,
            zrange=[vmin,vmax],
            masked_array=new_mask,
            mask_cmap=mask_cmap,
            savefig_dpi=400,
            axes=False
        ).save_figure()

    if "highpass" in image_set:
        high_pass = image_object.high_pass
        vmin = image_object.high_pass.min()
        vmax = image_object.high_pass.max()
        Images(
            data=high_pass,
            output_dir=save_dir,
            filename=f"{filename}_highpass",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highpass",
            cmap=cmap,
            zrange=[vmin,vmax],
            axes=False
        ).save_figure()

    if "lowpass" in image_set:
        low_pass = image_object.low_pass
        vmin=image_object.low_pass.min()
        vmax=image_object.low_pass.max()
        Images(
            data=low_pass,
            output_dir=save_dir,
            filename=f"{filename}_lowpass",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Lowpass",
            cmap=cmap,
            zrange=[vmin,vmax],
            axes=False
        ).save_figure()

    if "mask" in image_set:
        Images(
            data=new_mask,
            output_dir=save_dir,
            filename=f"{filename}_mask",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Binary Mask",
            cmap=cmap,
            image_type="binary",
            axes=False
        ).save_figure()

    # if "numbered_grains" in image_set:
    #     # Create the original image with the mask overlayed
    #     original = image_object.image_original
    #     vmin = original.min()
    #     vmax = original.max()
    #     Images(
    #         data=original,
    #         output_dir=save_dir,
    #         filename=f"{filename}_lowpass.png",
    #         pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
    #         title="Highpass with mask",
    #         cmap=cmap,
    #         zrange=[vmin,vmax],
    #     ).save_figure()

    #     # Create the high passed image with the mask overlayed
    #     high_pass = image_object.high_pass
    #     highmin = high_pass.min()
    #     highmax = high_pass.max()
    #     norm_highpass = normalise_array(high_pass)
    #     rgba_highpass = get_cmap(norm_highpass)
    #     rgb_highpass = rgba_highpass[..., :3]
    #     rgb_highpass[new_mask > 0] = [0, 0, 1]

    #     # Create the figure and size appropriately to avoid anti-aliasing
    #     dpi = 100
    #     height, width, _ = rgb_original.shape
    #     colourbar_width = 20
    #     gap = 40
    #     total_width = width + gap + colourbar_width
    #     orig_fig = plt.figure(figsize=(total_width / dpi, height / dpi), dpi=dpi)
    #     high_fig = plt.figure(figsize=(total_width / dpi, height / dpi), dpi=dpi)
    #     ax_orig = orig_fig.add_axes([0,0,1,1])
    #     ax_orig.set_title("Numbered grains (original)")
    #     ax_orig.imshow(rgb_original, cmap='afmhot', interpolation='nearest')
    #     ax_orig.axis('off')
    #     ax_high = high_fig.add_axes([0,0,1,1])
    #     ax_high.set_title("Numbered grains (highpass)")
    #     ax_high.imshow(rgb_highpass, cmap='afmhot', interpolation='nearest')
    #     ax_high.axis('off')

    #     # Place numbers on their respective grains
    #     outline = [path_effects.withStroke(linewidth=1, foreground='black')]
    #     for grain in image_object.grains.values():
    #         id = grain.grain_id
    #         (x, y) = grain.grain_centre_coords
    #         offset = (0,0)
    #         ax_orig.text(y+offset[0], x+offset[1], id, color="white", fontsize=3, ha="center", va="center", path_effects=outline)
    #         ax_high.text(y+offset[0], x+offset[1], id, color="white", fontsize=3, ha="center", va="center", path_effects=outline)

    #     orig_fig.savefig(save_dir / f"{filename}_original_numbered.png", bbox_inches="tight", pad_inches=0.1, dpi=dpi*4)
    #     plt.close(orig_fig)
    #     high_fig.savefig(save_dir / f"{filename}_highpass_numbered.png", bbox_inches="tight", pad_inches=0.1, dpi=dpi*4)
    #     plt.close(high_fig)

    if "original_mask" in image_set:
        # Save original image with mask overlay
        original = image_object.image_original
        vmin = original.min()
        vmax = original.max()
        Images(
            data=original,
            output_dir=save_dir,
            filename=f"{filename}_original_mask",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Original Image with Mask",
            cmap=cmap,
            zrange=[vmin,vmax],
            masked_array=new_mask,
            mask_cmap=mask_cmap,
            axes=False,
            savefig_dpi=400,
        ).save_figure()

    if "original" in image_set:
        original = image_object.image_original
        vmin = original.min()
        vmax = original.max()
        Images(
            data=original,
            output_dir=save_dir,
            filename=f"{filename}_original",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Original Image",
            cmap=cmap,
            zrange=[vmin,vmax],
            axes=False
        ).save_figure()

    if "rgb_grains" in image_set:
        Images(
            data=image_object.mask_rgb,
            output_dir=save_dir,
            filename=f"{filename}_rgb_grains",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Coloured Grains",
            axes=False,
            colorbar=False
        ).save_figure()

    if "smears" in image_set:
        norm_smears = normalise_array(image_object.high_pass)
        rgba_smears = get_cmap(norm_smears)
        rgba_smears = rgba_smears[..., :3]
        mask_2d = np.all(image_object.mask_rgb == 0, axis=2)
        rgba_smears[mask_2d == 0] = [1, 1, 1]
        rgba_smears[image_object.smears == 1] = [1, 0, 1]

        Images(
            data=rgba_smears,
            output_dir=save_dir,
            filename=f"{filename}_smears",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highlighted Smears",
            cmap=cmap,
            zrange=[vmin,vmax],
            axes=False
        ).save_figure()

    logger.info(f"[{filename}] : Saved {len(image_set)} images to {save_dir}")


def save_image(
        image: np.ndarray,
        output_dir: Path,
        filename: str,
        cmap: str = 'afmhot',
        vmin: float = None,
        vmax: float = None,
        pixel_to_nm_scaling: float = None
    ) -> None:
    """
    Save an array to file as an image

    Parameters
    ----------
    image : np.ndarray
        The array to save to file.
    output_dir : Path
        The directory to save the image to.
    filename : str
        The name of the file to be created/ saved to.
    cmap : str
        The cmap to be used in the imsave() function defined in config. By default is 'grey'.
    vmin : float
        The minimum height value of the image before normalisation.
    vmax : float
        The maximum height value of the image before normalisation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if pixel_to_nm_scaling:
        nm_min = vmin / pixel_to_nm_scaling
        nm_max = vmax / pixel_to_nm_scaling

        fig, ax = plt.subplots()
        image_norm = normalise_array(image)
        im = ax.imshow(image_norm, cmap=cmap, vmin=0, vmax=1)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Height (nm)")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([f"{v:.2f}" for v in np.linspace(nm_min, nm_max, 5)])

        ax.axis("off")
        fig.savefig(output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.imsave(output_dir / filename, image, cmap=cmap)


def save_to_csv(df: pd.DataFrame, output_filename: str) -> None:
    """
    Method for saving pd.DataFrames to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The created dataframe to be saved.
    output_filename : str
        The output directory and filename.
    """
    df.to_csv(output_filename, index=False)


def save_config(config: dict, output_filename: str) -> None:
    """
    Method for saving config options to .yaml.

    Parameters
    ----------
    config : dict
        Dictionary containing all config options to be saved.
    output_filename: str
        The output directory and filename.
    """
    with (output_filename).open("w") as outfile:
        safe_dump(config, outfile, default_flow_style=False)


def grain_area_histogram(data: list[float], filename: str, output_dir: Path):
    """
    Method for saving a histogram plotting the areas of grains found.

    Parameters
    ----------
    data : List[float]
        A list of the datapoints (areas) for each grain.
    filename : str
        The name of the file being processed.
    output_dir : Path
        The main directory outputs are saved to.
    """
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data, bins='auto', kde=True, log_scale=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Grain areas nm²')
        plt.tight_layout()
        full_output_dir = output_dir / "graphs"
        full_output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_output_dir / f"{filename}_grain_areas_hist.png", dpi=300)
        plt.close(fig)


def grain_circularity_histogram(data: list[float], filename: str, output_dir):
    """
    Method for saving a histogram plotting the circularity rating of grains found.

    Parameters
    ----------
    data : List[float]
        A list of the datapoints (circularity rating) for each grain.
    filename : str
        The name of the file being processed.
    output_dir : Path
        The main directory outputs are saved to.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, bins='auto', kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Grain circularities (0-1)')
    plt.tight_layout()
    full_output_dir = output_dir / "graphs"
    full_output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(full_output_dir / f"{filename}_grain_circularity_hist.png", dpi=300)
    plt.close(fig)

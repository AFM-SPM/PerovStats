from __future__ import annotations
from importlib import resources
from pathlib import Path
import math

from yaml import safe_dump
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
import seaborn as sns
from skimage import morphology
from skimage.measure import regionprops
import topostats

from .classes import ImageData
from .image_processing import normalise_array


def load_mplstyle(style: str | Path) -> None:
    """
    Load the Matplotlibrc parameter file.

    Parameters
    ----------
    style : str | Path
        Path to a Matplotlib Style file.
    """
    if style == "topostats.mplstyle":
        plt.style.use(resources.files(topostats) / style)
    else:
        plt.style.use(style)


class Images:
    def __init__(
        self,
        data: npt.NDArray,
        output_dir: str | Path,
        filename: str,
        title: str,
        pixel_to_nm_scaling: float,
        font_size: float,
        mask_data: npt.NDArray | None = None,
        number_grains: bool = False,
        use_scalebar: bool = False,
        cmap: str | None = None,
        region_properties: dict = None,
        savefig_format: str = "png",
        axes: bool = False,
        dpi: int | None = None,
        smears: np.ndarray | None = None,
    ):
        self.data = data
        self.mask_data = mask_data
        self.output_dir = output_dir
        self.filename = filename
        self.title = title
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.font_size = font_size
        self.number_grains = number_grains
        self.use_scalebar = use_scalebar
        self.cmap = cmap
        self.region_properties = region_properties
        self.savefig_format = savefig_format
        self.axes = axes
        self.dpi = dpi if dpi else None
        self.smears = smears


    def save_figure(self):
        fig, ax = plt.subplots(1, 1)

        # Add appropriate scalebar to image
        if self.use_scalebar:
            # Determine size of scalebar in image by taking 1/5 of the width and rounding to
            # the nearest suitable number (0.5, 1, 10, 100, etc.)
            scalebar = ScaleBar(
                self.pixel_to_nm_scaling,
                units="nm",
                box_alpha=0.9,
                location="lower right",
                font_properties={"size": self.font_size},
            )
            ax.add_artist(scalebar)

        # Add base data to image
        ax.imshow(self.data, cmap=self.cmap)

        # Add mask skeleton overlay if given
        if self.mask_data is not None:
            mask = np.ma.masked_where(self.mask_data == 0, self.mask_data)
            ax.imshow(
                mask,
                cmap="bwr"
            )

        # Add smear mask if given
        if self.smears is not None:
            smear_mask = np.ma.masked_where(self.smears == 0, self.smears)
            ax.imshow(
                smear_mask,
                cmap="spring"
            )

        plt.title(self.title, fontsize=self.font_size)
        plt.xlabel("Nanometres", fontsize=self.font_size)
        plt.ylabel("Nanometres", fontsize=self.font_size)
        plt.axis(self.axes)

        # Add grain numbers to the image, requires the region_properties dict to be passed into the class.
        # This will also mean calculations will have to be done for interpolation to avoid blurry edges in the data.
        if self.number_grains:
            fig, ax = number_grain_plots(
                fig,
                ax,
                self.region_properties
            )

        # Calulate minimum suitable dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            (self.output_dir / f"{self.filename}.{self.savefig_format}"),
            bbox_inches="tight",
            dpi=self.dpi
        )

        plt.close()


    def plot_histogram_and_save(self) -> tuple | None:
        """
        Plot and save a histogram of the height map.

        Returns
        -------
        tuple | None
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
        """
        if "all" in self.image_set:
            fig, ax = plt.subplots(1, 1)

            ax.hist(self.data.flatten().astype(float), bins=self.histogram_bins, log=self.histogram_log_axis)
            ax.set_xlabel("pixel height")
            if self.histogram_log_axis:
                ax.set_ylabel("frequency in image (log)")
            else:
                ax.set_ylabel("frequency in image")
            plt.title(self.title)
            plt.savefig(
                (self.output_dir / f"{self.filename}_histogram.{self.savefig_format}"),
                bbox_inches="tight",
                pad_inches=0.5,
                dpi=self.savefig_dpi,
            )
            plt.close()

            return fig, ax
        return None


def set_n_ticks(ax: plt.Axes.axes, n_xy: list[int | None, int | None]) -> None:
    """
    Set the number of ticks along the y and x axes and lets matplotlib assign the values.

    Parameters
    ----------
    ax : plt.Axes.axes
        The axes to add ticks to.
    n_xy : list[int, int]
        The number of ticks.

    Returns
    -------
    plt.Axes.axes
        The axes with the new ticks.
    """
    if n_xy[0] is not None:
        xlim = ax.get_xlim()
        xstep = (max(xlim) - min(xlim)) / (n_xy[0] - 1)
        xticks = np.arange(min(xlim), max(xlim) + xstep, xstep)
        ax.set_xticks(np.round(xticks))
    if n_xy[1] is not None:
        ylim = ax.get_ylim()
        ystep = (max(ylim) - min(ylim)) / (n_xy[1] - 1)
        yticks = np.arange(min(ylim), max(ylim) + ystep, ystep)
        ax.set_yticks(np.round(yticks))


def number_grain_plots(
    fig, ax, region_properties: list
) -> tuple:
    """
    Add the grain numbers to the plot.

    Parameters
    ----------
    fig : plt.figure.Figure
        Matplotlib.pyplot figure object.
    ax : plt.axes._subplots.AxesSubplot
        Matplotlib.pyplot axes object.
    shape : tuple
        Tuple of the image-to-be-plot's shape.
    region_properties : list
        Region properties to add bounding boxes from.
    pixel_to_nm_scaling : float
        The scaling factor from px to nm.

    Returns
    -------
    tuple
        Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
    """
    # Select an appropriate font size - as large as possible without significantly obscuring grains
    # or overlapping with other numbers.
    # Calculations use pixels here.
    grain_areas = np.array([x['area'] for x in region_properties])
    # Remove outlier values by z-score and find average
    mean = grain_areas.mean()
    std = grain_areas.std()
    filtered_areas = grain_areas[np.abs((grain_areas - mean) / std) < 3]
    avr_grain_area = filtered_areas.mean()

    font_size = math.ceil(avr_grain_area / 35000)

    # blit each grain's id onto its centre point in the image
    for i, region in enumerate(region_properties):
        # Place the number in the centre of its grain
        min_y, min_x, max_y, max_x = (x for x in region['bbox'])
        x_loc = min_x + (0.5 * (max_x - min_x))
        y_loc = min_y + (0.5 * (max_y - min_y))
        # Number (white)
        numbering = ax.text(x_loc, y_loc, i, fontsize=font_size, color="white", ha="center", va="center")
        # Border (black)
        numbering.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"), path_effects.Normal()])

    return fig, ax


def save_images(config: dict[str, any], image_object: ImageData, variation: str=None) -> None:
    output_config = config["output"]
    cmap = output_config["colour_scheme"]
    number_grains = output_config["number_grains"]
    image_set = output_config["image_set"]
    scalebar = output_config["scalebar"]
    font_size = output_config["font_size"]
    output_dir = Path(config["output_dir"])

    filename = image_object.filename
    file_output_dir = Path(output_dir / filename)
    file_output_dir.mkdir(parents=True, exist_ok=True)
    if variation:
        save_dir = file_output_dir / "images" / variation
    else:
        save_dir = file_output_dir / "images"

    # Remove mask sections that overlap with smear areas
    new_mask = image_object.mask.copy()
    new_mask[image_object.edge_grains] = 0
    if config["remove_smears"]["run"]:
        new_mask[image_object.smear_grains] = 0
    # Remove single pixels left in the smear area by accident
    new_mask = morphology.remove_small_objects(new_mask, max_size=1, connectivity=2)
    image_object.cleaned_mask = new_mask

    # Placeholder dpi
    dpi = image_object.image_original.shape[0]

    # Collect grain info needed for grain numbering (if applicable)
    if number_grains:
        mask_regionprops = [{"id": grain.grain_id, "bbox": grain.grain_bbox, "area": grain.grain_area} for grain in image_object.grains.values()]
    else:
        mask_regionprops = []

    # Create each image defined in image_set, adding mask overlays and grain numbers where appropriate.
    # This includes calculating a suitable interpolation value to avoid blurring of any part without making the
    # image unnecessarily large.
    if "highpass_mask" in image_set:
        Images(
            data=image_object.high_pass,
            mask_data=image_object.cleaned_mask,
            output_dir=save_dir,
            filename=f"{filename}_highpass_mask",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highpass with mask",
            cmap=cmap,
            number_grains=number_grains,
            region_properties=mask_regionprops,
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "highpass" in image_set:
        Images(
            data=image_object.high_pass,
            output_dir=save_dir,
            filename=f"{filename}_highpass",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highpass",
            cmap=cmap,
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "lowpass" in image_set:
        Images(
            data=image_object.low_pass,
            output_dir=save_dir,
            filename=f"{filename}_lowpass",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Lowpass",
            cmap=cmap,
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "original_mask" in image_set:
        Images(
            data=image_object.image_original,
            mask_data=image_object.cleaned_mask,
            output_dir=save_dir,
            filename=f"{filename}_original_mask",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Original with mask",
            cmap=cmap,
            number_grains=number_grains,
            region_properties=mask_regionprops,
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "original" in image_set:
        Images(
            data=image_object.image_original,
            output_dir=save_dir,
            filename=f"{filename}_original",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Original",
            cmap=cmap,
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "rgb_grains" in image_set:
        Images(
            data=image_object.mask_rgb,
            output_dir=save_dir,
            filename=f"{filename}_rgb_grains",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Coloured Grains",
            dpi=dpi,
            use_scalebar=scalebar,
            font_size=font_size,
        ).save_figure()

    if "smears" in image_set:
        Images(
            data=image_object.high_pass,
            mask_data=image_object.cleaned_mask,
            output_dir=save_dir,
            filename=f"{filename}_smears",
            pixel_to_nm_scaling=image_object.pixel_to_nm_scaling,
            title="Highlighted Smears",
            cmap=cmap,
            dpi=dpi,
            use_scalebar=scalebar,
            smears=image_object.smears,
            font_size=font_size,
        ).save_figure()


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


def get_region_properties(image: npt.NDArray, **kwargs) -> list:
        """
        Extract the properties of each region.

        Parameters
        ----------
        image : np.array
            Numpy array representing image.
        **kwargs :
            Arguments passed to 'skimage.measure.regionprops(**kwargs)'.

        Returns
        -------
        list
            List of region property objects.
        """
        return regionprops(image, **kwargs)


def label_regions(image: npt.NDArray, background: int = 0) -> npt.NDArray:
        """
        Label regions.

        This method is used twice, once prior to removal of small regions and again afterwards which is why an image
        must be supplied rather than using 'self'.

        Parameters
        ----------
        image : npt.NDArray
            2-D Numpy array of image.
        background : int
            Value used to indicate background of image. Default = 0.

        Returns
        -------
        npt.NDArray
            2-D Numpy array of image with regions numbered.
        """
        return morphology.label(image, background)

from __future__ import annotations
from importlib import resources
from pathlib import Path

from yaml import safe_dump
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
from skimage import morphology
from skimage.measure import regionprops
from loguru import logger
import topostats
from topostats.theme import Colormap

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
    """
    Plots image arrays.

    Parameters
    ----------
    data : npt.NDArray
        Numpy array to plot.
    output_dir : str | Path
        Output directory to save the file to.
    filename : str
        Filename to save image as.
    style : str | Path
        Filename of matplotlibrc parameters.
    pixel_to_nm_scaling : float
        The scaling factor showing the real length of 1 pixel in nanometers (nm).
    masked_array : npt.NDArray
        Optional mask array to overlay onto an image.
    plot_coords : npt.NDArray
        ??? Needs defining.
    title : str
        Title for plot.
    image_type : str
        The image data type, options are 'binary' or 'non-binary'.
    module : str
            The name of the module plotting the images.
    image_set : str
        The set of images to process, options are 'core' or 'all'.
    core_set : bool
        Flag to identify image as part of the core image set or not.
    pixel_interpolation : str, optional
        Interpolation to use (default is 'None').
    grain_crop_plot_size_nm : float, optional
        Size in nm of the square cropped grain images if using the grains image set. If -1,
        will use the grain's default bounding box size.
    cmap : str, optional
        Colour map to use (default 'nanoscope', 'afmhot' also available).
    mask_cmap : str
        Colour map to use for the secondary (masked) data (default 'jet_r', 'blu' provides more contrast).
    region_properties : dict
        Dictionary of region properties, adds bounding boxes if specified.
    zrange : list
        Lower and upper bound to clip core images to.
    colorbar : bool
        Optionally add a colorbar to plots, default is False.
    axes : bool
        Optionally add/remove axes from the image.
    num_ticks : tuple[int | None]
        The number of x and y ticks to display on the iage.
    save : bool
        Whether to save the image.
    savefig_format : str, optional
        Format to save the image as.
    histogram_log_axis : bool
        Optionally use a loagrithmic y-axis for the histogram plots.
    histogram_bins : int, optional
        Number of bins for histograms to use.
    savefig_dpi : str | float, optional
        The resolution of the saved plot (default 'figure').
    number_grains : bool
        Optionally number each grain in a plot.
    trace_linewidth : float
        Width of lines when plotting splines/curvature.
    pad_inches : float
        Inches to pad a figure by to allow for heightscales + names.
    """

    def __init__(
        self,
        data: npt.NDArray,
        output_dir: str | Path,
        filename: str,
        style: str | Path = None,
        pixel_to_nm_scaling: float = 1.0,
        masked_array: npt.NDArray = None,
        plot_coords: npt.NDArray = None,
        title: str = None,
        image_type: str = "non-binary",
        module: str = "",
        image_set: str = "core",
        core_set: bool = False,
        pixel_interpolation: str | None = None,
        grain_crop_plot_size_nm: float = -1,
        cmap: str | None = None,
        mask_cmap: str = "jet_r",
        region_properties: dict = None,
        zrange: list = None,
        colorbar: bool = True,
        axes: bool = True,
        num_ticks: tuple[int | None] = (None, None),
        save: bool = True,
        savefig_format: str | None = None,
        histogram_log_axis: bool = True,
        histogram_bins: int | None = None,
        savefig_dpi: str | float | None = None,
        number_grains: bool = False,
        trace_linewidth: float = 1.0,
        pad_inches: float | None = None,
    ) -> None:
        """
        Initialise the class.

        There are two key parameters that ensure whether an image is plotted that are passed in from the updated
        plotting dictionary. These are the ``image_set`` which defines which images to plot. ``all`` images plots
        everything, or ``core`` only plots the core set.
        There is then the 'core_set' which defines whether an individual images belongs to the 'core_set' or
        not. If it doesn't then it is not plotted when `image_set` is `["core"]`.

        Parameters
        ----------
        data : npt.NDArray
            Numpy array to plot.
        output_dir : str | Path
            Output directory to save the file to.
        filename : str
            Filename to save image as.
        style : str | Path
            Filename of matplotlibrc parameters.
        pixel_to_nm_scaling : float
            The scaling factor showing the real length of 1 pixel in nanometers (nm).
        masked_array : npt.NDArray
            Optional mask array to overlay onto an image.
        plot_coords : npt.NDArray
            ??? Needs defining.
        title : str
            Title for plot.
        image_type : str
            The image data type, options are 'binary' or 'non-binary'.
        module : str
            The name of the module plotting the images.
        image_set : str
            The set of images to process, options are 'core' or 'all'.
        core_set : bool
            Flag to identify image as part of the core image set or not.
        pixel_interpolation : str, optional
            Interpolation to use (default is 'None').
        grain_crop_plot_size_nm : float, optional
            Size in nm of the square cropped grain images if using the grains image set. If -1,
            will use the grain's default bounding box size.
        cmap : str, optional
            Colour map to use (default 'nanoscope', 'afmhot' also available).
        mask_cmap : str
            Colour map to use for the secondary (masked) data (default 'jet_r', 'blu' provides more contrast).
        region_properties : dict
            Dictionary of region properties, adds bounding boxes if specified.
        zrange : list
            Lower and upper bound to clip core images to.
        colorbar : bool
            Optionally add a colorbar to plots, default is False.
        axes : bool
            Optionally add/remove axes from the image.
        num_ticks : tuple[int | None]
            The number of x and y ticks to display on the iage.
        save : bool
            Whether to save the image.
        savefig_format : str, optional
            Format to save the image as.
        histogram_log_axis : bool
            Optionally use a loagrithmic y-axis for the histogram plots.
        histogram_bins : int, optional
            Number of bins for histograms to use.
        savefig_dpi : str | float, optional
            The resolution of the saved plot (default 'figure').
        number_grains : bool
            Optionally number each grain in a plot.
        trace_linewidth : float
            Width of lines when plotting splines/curvature.
        pad_inches : float
            Inches to pad a figure by to allow for heightscales + names.
        """
        if style is None:
            style = "topostats.mplstyle"
        load_mplstyle(style)
        if zrange is None:
            zrange = [None, None]
        self.data = data
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.pixel_to_nm_scaling = pixel_to_nm_scaling
        self.masked_array = masked_array
        self.plot_coords = plot_coords
        self.title = title
        self.image_type = image_type
        self.module = module
        self.image_set = image_set
        self.core_set = core_set
        self.interpolation = mpl.rcParams["image.interpolation"] if pixel_interpolation is None else pixel_interpolation
        cmap = mpl.rcParams["image.cmap"] if cmap is None else cmap
        self.cmap = Colormap(cmap).get_cmap()
        self.mask_cmap = Colormap(mask_cmap).get_cmap()
        self.region_properties = region_properties
        self.zrange = zrange
        self.colorbar = colorbar
        self.axes = axes
        self.num_ticks = num_ticks
        self.save = save
        self.savefig_format = mpl.rcParams["savefig.format"] if savefig_format is None else savefig_format
        self.histogram_log_axis = histogram_log_axis
        self.histogram_bins = mpl.rcParams["hist.bins"] if histogram_bins is None else histogram_bins
        self.savefig_dpi = mpl.rcParams["savefig.dpi"] if savefig_dpi is None else savefig_dpi
        self.number_grains = number_grains
        self.trace_linewidth = trace_linewidth
        self.pad_inches = pad_inches if pad_inches is not None else mpl.rcParams["savefig.pad_inches"]

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

    def save_figure(self):
        """
        Save figures as plt.savefig objects.

        Returns
        -------
        tuple
            Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
        """
        fig, ax = plt.subplots(1, 1)
        shape = self.data.shape
        if isinstance(self.data, np.ndarray):
            im = ax.imshow(
                self.data,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=self.cmap,
                vmin=self.zrange[0],
                vmax=self.zrange[1],
            )
            if isinstance(self.masked_array, np.ndarray):
                mask = np.ma.masked_where(self.masked_array == 0, self.masked_array)
                ax.imshow(
                    mask,
                    cmap=self.mask_cmap,
                    extent=(
                        0,
                        shape[1] * self.pixel_to_nm_scaling,
                        0,
                        shape[0] * self.pixel_to_nm_scaling,
                    ),
                    interpolation=self.interpolation,
                    alpha=0.7,
                )
                patch = [Patch(color=self.mask_cmap(1, 0.7), label="Mask")]
                plt.legend(handles=patch, loc="upper right", bbox_to_anchor=(1.02, 1.09))
            # If coordinates are provided (such as in splines), plot those. These can be in two forms, a list of numpy
            # arrays for each grain (if plotting whole image) or a single list if plotting grain/molecule
            elif self.plot_coords is not None:
                # If self.plot_coords is a numpy array then we are plotting on a grain/molecule basis and have nothing
                # to loop over so addit to a list so we can loop over it
                if isinstance(self.plot_coords, np.ndarray):
                    self.plot_coords = [self.plot_coords]
                for grain_coords in self.plot_coords:
                    ax.plot(
                        grain_coords[:, 1] * self.pixel_to_nm_scaling,
                        (shape[0] - grain_coords[:, 0]) * self.pixel_to_nm_scaling,
                        c="c",
                        linewidth=self.trace_linewidth,
                    )

            plt.title(self.title)
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            set_n_ticks(ax, self.num_ticks)
            plt.axis(self.axes)
            if self.colorbar and self.image_type == "non-binary":
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, label="Height (Nanometres)")
            if self.region_properties:
                if self.number_grains:
                    fig, ax = number_grain_plots(
                        fig,
                        ax,
                        shape,
                        self.region_properties,
                        self.pixel_to_nm_scaling,
                        (2, -2),
                    )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not self.axes and not self.colorbar:
                plt.title("")
                fig.frameon = False
                plt.box(False)
                plt.tight_layout()
                plt.savefig(
                    (self.output_dir / f"{self.filename}.{self.savefig_format}"),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=self.savefig_dpi,
                )
            else:
                plt.savefig(
                    (self.output_dir / f"{self.filename}.{self.savefig_format}"),
                    bbox_inches="tight",
                    dpi=self.savefig_dpi,
                )
        else:
            plt.xlabel("Nanometres")
            plt.ylabel("Nanometres")
            self.data.show(
                ax=ax,
                extent=(0, shape[1] * self.pixel_to_nm_scaling, 0, shape[0] * self.pixel_to_nm_scaling),
                interpolation=self.interpolation,
                cmap=self.cmap,
            )
        plt.close()
        return fig, ax


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


# def add_bounding_boxes_to_plot(fig, ax, shape: tuple, region_properties: list, pixel_to_nm_scaling: float) -> tuple:
#     """
#     Add the bounding boxes to a plot.

#     Parameters
#     ----------
#     fig : plt.figure.Figure
#         Matplotlib.pyplot figure object.
#     ax : plt.axes._subplots.AxesSubplot
#         Matplotlib.pyplot axes object.
#     shape : tuple
#         Tuple of the image-to-be-plot's shape.
#     region_properties : list
#         Region properties to add bounding boxes from.
#     pixel_to_nm_scaling : float
#         The scaling factor from px to nm.

#     Returns
#     -------
#     tuple
#         Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
#     """
#     for region in region_properties:
#         min_y, min_x, max_y, max_x = (x * pixel_to_nm_scaling for x in region['bbox'])
#         # Correct y-axis
#         min_y = (shape[0] * pixel_to_nm_scaling) - min_y
#         max_y = (shape[0] * pixel_to_nm_scaling) - max_y
#         rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="white", linewidth=0)
#         ax.add_patch(rectangle)
#     return fig, ax


def number_grain_plots(
    fig, ax, shape: tuple, region_properties: list, pixel_to_nm_scaling: float, offset: tuple
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
    offset : tuple
        The amount to shift the number to avoid overlap with bounding boxes (x, y).

    Returns
    -------
    tuple
        Matplotlib.pyplot figure object and Matplotlib.pyplot axes object.
    """
    for i, region in enumerate(region_properties):
        min_y, min_x, max_y, max_x = (x * pixel_to_nm_scaling for x in region['bbox'])
        # Correct y-axis
        min_y = (shape[1] * pixel_to_nm_scaling) - min_y
        max_y = (shape[1] * pixel_to_nm_scaling) - max_y
        x_loc = min_x + (0.5 * (max_x - min_x)) + offset[0]
        y_loc = min_y + (0.5 * (max_y - min_y)) + offset[1]
        numbering = ax.text(x_loc, y_loc, i, fontsize=3, color="white", ha="center", va="center")
        numbering.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"), path_effects.Normal()])
    return fig, ax


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
    cmap = config["output"]["colour_scheme"]
    number_grains = config["output"]["number_grains"]
    colorbar = config["output"]["height_scale"]
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

        mask_regionprops = [{"id": grain.grain_id, "bbox": grain.grain_bbox} for grain in image_object.grains.values()]

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
            axes=False,
            pad_inches=20,
            number_grains=number_grains,
            region_properties=mask_regionprops,
            colorbar=colorbar
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
            axes=False,
            colorbar=colorbar
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
            axes=False,
            colorbar=colorbar
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
        if config["output"]["height_scale"]:
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
                pad_inches=20,
                number_grains=number_grains,
                region_properties=mask_regionprops,
                colorbar=colorbar
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
            axes=False,
            colorbar=colorbar
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
            axes=False,
            colorbar=colorbar
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

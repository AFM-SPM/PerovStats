from __future__ import annotations
from pathlib import Path

from loguru import logger
import numpy.typing as npt
from skimage import morphology
from skimage.measure import regionprops
from scipy.special import erf
import skimage as ski
import numpy as np
from matplotlib import pyplot as plt

from .classes import ImageData
from .image_processing import normalise_array


def segment_image(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    """
    Main method for splitting an image by frequency.

    Parameters
    ----------
    config: dict[str, any]
        Dictionary of configuration settings
    image_object
        Class object of the current image containing all relevant
        data.
    """
    output_dir = Path(config["output_dir"])

    if image_object.high_pass is not None:
        # For each image create and save a mask
        fname = image_object.filename
        im = image_object.high_pass
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling

        # Scale threshold block size with image scaling and round to nearest odd integer
        threshold_block_size = config["segmentation"]["threshold_block_size"] / pixel_to_nm_scaling
        threshold_block_size = 2 * round((threshold_block_size - 1) / 2) + 1

        # Cleaning config options - adjusted for pixel to nm scaling
        area_threshold = config["segmentation"]["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = config["segmentation"]["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options - adjusted for pixel to nm scaling
        smooth_sigma = config["segmentation"]["smoothing"]["sigma"]
        if smooth_sigma:
            smooth_sigma = smooth_sigma / pixel_to_nm_scaling

        logger.info(f"[{image_object.filename}] : *** Mask creation ***")
        logger.info(f"[{image_object.filename}] : Creating grain mask")
        np_mask = create_grain_mask(
            im,
            threshold_block_size=threshold_block_size,
            smooth_sigma=smooth_sigma,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
        )

        image_object.mask = np_mask

        # Convert to image format and save
        img_dir = Path(output_dir) / fname / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask.jpg", np_mask)

        # Save high-pass with mask skeleton
        high_pass = image_object.high_pass
        rgb_highpass = np.stack((high_pass,)*3, axis=-1)
        rgb_highpass = normalise_array(rgb_highpass)
        rgb_highpass[np_mask > 0] = [1, 0, 0]
        plt.imsave(output_dir / fname / "images" / f"{fname}_mask_overlay.jpg", rgb_highpass)



def clean_mask(
    mask: np.ndarray,
    area_threshold: float = 100,
    disk_radius: int = 4,
) -> np.ndarray:
    """
    Clean up grain mask by connecting close segments and removing small sections.

    Parameters
    ----------
    mask : np.ndarray
        Mask array.
    area_threshold : float, optional
        Area threshold for cleaning up mask.
    disk_radius : int, optional
        Disk radius for cleaning up mask.

    Returns
    -------
    numpy.ndarray
        Cleaned up mask array.
    """
    mask = ski.morphology.remove_small_holes(
        ski.morphology.remove_small_objects(mask, max_size=area_threshold)
    )
    return ski.morphology.opening(mask, ski.morphology.disk(disk_radius))


def create_grain_mask(
    im: np.ndarray,
    threshold_block_size: float,
    smooth_sigma: float,
    area_threshold: float,
    disk_radius: float,
) -> np.ndarray:
    """
    Use local thresholding to find grain edges and create a skeletonised mask of
    borders.

    Parameters
    ----------
    im : np.ndarray
        Image to be masked.
    threshold_block_size : float
        Size of blocks to be thresholded once at a time.
    smooth_sigma : float
        Amount of smoothing applied to the image before thresholding.
    area_threshold : float
        Maximum size of a grain considered too small to count.
    disk_radius : float
        How far to look for closeby segments when connecting them.

    Returns
    -------
    np.ndarray
        Skeletonised mask of grain borders.
    """
    im_ = ski.filters.gaussian(im, sigma=smooth_sigma) # Smooth image
    local_thresh = ski.filters.threshold_local(im_, block_size=threshold_block_size) # Gets an array of local thresholds
    mask = im_ > local_thresh
    mask = clean_mask(mask, area_threshold, disk_radius) if area_threshold else mask
    selection = ski.util.invert(mask)
    return ski.morphology.skeletonize(selection)


def apply_cutoff(
    f_grid,
    cutoff: float,
    edge_width: float = 0,
) -> np.ndarray:
    """
    Create a mask to filter frequencies.

    Parameters
    ----------
    f_grid : np.ndarray
        Frequency grid of the image, each pixel containing a value indicating
        the distance from the zero-frequency component in a radial frequency map.
    cutoff : float
        The spatial frequency cut off, expressed as a relative fraction
        of the Nyquist frequency.
    edge_width : float
        Edge width, expressed as a relative fraction of the Nyquist
        frequency. If zero, the filter has sharp edges. For non-zero
        values the transition has the shape of the error function,
        with the specified width.

    Returns
    -------
    np.ndarray
        Frequency mask.
    """
    if edge_width and edge_width > 0:
        return 0.5 * (erf((f_grid - cutoff) / edge_width) + 1)
    else:
        return (f_grid >= cutoff).astype(np.float64)


@staticmethod
def tidy_border(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Remove whole grains touching the border.

    Parameters
    ----------
    mask : npt.NDArray
        3-D Numpy array of the grain mask tensor.

    Returns
    -------
    npt.NDArray
        3-D Numpy array of the grain mask tensor with grains touching the border removed.
    """
    # Find the grains that touch the border then remove them from the full mask tensor
    mask_labelled = morphology.label(mask)
    mask_regionprops = regionprops(mask_labelled)
    for region in mask_regionprops:
        if (
            region.bbox[0] == 0
            or region.bbox[1] == 0
            or region.bbox[2] == mask.shape[0]
            or region.bbox[3] == mask.shape[1]
        ):
            mask[mask_labelled == region.label] = 0

    return mask


def create_frequency_mask(image: np.ndarray) -> np.ndarray:
    # Create frequency mask grid
    yres, xres = image.shape
    xr = np.arange(xres)
    yr = np.arange(yres)
    fx = 2 * np.fmin(xr, xres - xr) / xres
    fy = 2 * np.fmin(yr, yres - yr) / yres

    # Full coordinate arrays
    xx, yy = np.meshgrid(fx, fy)
    return np.sqrt(xx**2 + yy**2)

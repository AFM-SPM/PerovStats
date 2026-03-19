from __future__ import annotations
from pathlib import Path
import heapq

from loguru import logger
import numpy.typing as npt
from skimage import morphology
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
import skimage as ski
import numpy as np

from .io import save_image
from .classes import ImageData
from .image_processing import normalise_array


def segment_image(
    config: dict[str, any],
    image_object: ImageData
) -> None:
    """
    Create the segmentation mask for an image, ready to be analysed for grain finding.
    This method also saves images of the mask.

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

        config = config["segmentation"]

        # Scale threshold block size with image scaling and round to nearest odd integer
        threshold_block_size = config["threshold_block_size"] / pixel_to_nm_scaling
        threshold_block_size = 2 * round((threshold_block_size - 1) / 2) + 1

        threshold_offset = config["threshold_offset"]

        # Cleaning config options - adjusted for pixel to nm scaling
        area_threshold = config["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = config["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options - adjusted for pixel to nm scaling
        smooth_sigma = config["smoothing"]["sigma"]
        if smooth_sigma:
            smooth_sigma = smooth_sigma / pixel_to_nm_scaling

        # Skeletonisation config option
        height_bias = config["height_bias"]

        logger.info(f"[{image_object.filename}] : *** Mask creation ***")
        logger.info(f"[{image_object.filename}] : Creating grain mask")
        # Create the grain mask by thresholding the image and skeletonising the result
        np_mask = create_grain_mask(
            im,
            threshold_block_size=threshold_block_size,
            threshold_offset=threshold_offset,
            smooth_sigma=smooth_sigma,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
            height_bias=height_bias
        )

        image_object.mask = np_mask

        # Convert to image format and save
        img_dir = Path(output_dir) / fname / "images"
        save_image(np_mask, img_dir, f"{fname}_mask.jpg")

        # Save high-pass with mask skeleton
        high_pass = image_object.high_pass
        rgb_highpass = np.stack((high_pass,)*3, axis=-1)
        rgb_highpass = normalise_array(rgb_highpass)
        rgb_highpass[np_mask > 0] = [1, 0, 0]
        save_image(rgb_highpass, img_dir, f"{fname}_mask_overlay.jpg")


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
    threshold_offset: float,
    smooth_sigma: float,
    area_threshold: float,
    disk_radius: float,
    height_bias: float,
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
    threshold_offset : float
        Offset of the threshold calculated in threshold_local.
    smooth_sigma : float
        Amount of smoothing applied to the image before thresholding.
    area_threshold : float
        Maximum size of a grain considered too small to count.
    disk_radius : float
        How far to look for closeby segments when connecting them.
    height_bias : float
        How much to weight height over the centre of a line during skeletonisation.

    Returns
    -------
    np.ndarray
        Skeletonised mask of grain borders.
    """
    im_ = ski.filters.gaussian(im, sigma=smooth_sigma)
    # Get an array of thresholds for each pixel
    local_thresh = ski.filters.threshold_local(im_, block_size=threshold_block_size, offset=threshold_offset)
    mask = im_ > local_thresh
    mask = clean_mask(mask, area_threshold, disk_radius) if area_threshold else mask
    selection = ski.util.invert(mask)
    skeleton = Skeletonisation(im_, selection, height_bias=height_bias).do_skeletonisation()

    return skeleton


class Skeletonisation:
    """
    Skeletonise a binary array following Zhang's algorithm (Zhang and Suen, 1984).

    Modifications are made to the published algorithm during the removal step to remove a fraction of the smallest pixel
    values opposed to all of them in the aforementioned algorithm. All operations are performed on the mask entered.

    Parameters
    ----------
    image : npt.NDArray
        Original 2D image containing the height data.
    mask : npt.NDArray
        Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
    height_bias : float
        Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria. 1 is all pixels
        smiilar to Zhang.
    """
    def __init__(self, image: np.ndarray, mask: np.ndarray, height_bias: float = 0.6):
        """
        Initialise the class.

        Parameters
        ----------
        image : npt.NDArray
            Original 2D image containing the height data.
        mask : npt.NDArray
            Binary image containing the object to be skeletonised. Dimensions should match those of 'image'.
        height_bias : float
            Ratio of lowest intensity (height) pixels to total pixels fitting the skeletonisation criteria.
        """
        self.image = image
        self.mask = np.pad(mask.copy(), pad_width=1, mode='constant', constant_values=0)
        self.height_bias = height_bias


    def do_skeletonisation(self) -> np.ndarray:
        """
        Perform skeletonisation.

        Returns
        -------
        npt.NDArray
            The single pixel thick, skeletonised array.
        """
        priority_map = self.calculate_priority_map()
        self.skeletonise_with_bias(priority_map)

        self.mask = self.mask[1:-1, 1:-1] # Remove padding added in __init__() to handle edge pixels
        return ski.morphology.skeletonize(self.mask)


    def calculate_priority_map(self) -> np.ndarray:
        """
        Create an array of size mask.shape containing priority scores for each pixel.

        The scores are calculated with: score = distance_to_edge + (1.0 - normalised_height) * height_bias
        This means a higher height bias reduces the importance of the pixel being in the centre of the line
        being skeletonised.

        Returns
        -------
        np.ndarray
            The priority map for each pixel in the image. Pixels not part of the mask are marked as 0.
        """
        # Create array of shape mask.shape with score of distance from edge
        dist = distance_transform_edt(self.mask)

        padded_image = np.pad(self.image, pad_width=1, mode='edge')

        # Normalise the heightmap
        img_min, img_max = padded_image.min(), padded_image.max()
        norm_height = (padded_image - img_min) / (img_max - img_min + 1e-8)

        # Combine the two arrays - (1.0 - norm_height to delete lighter pixels)
        priority_map = dist + (1.0 - norm_height) * self.height_bias

        return priority_map


    def skeletonise_with_bias(self, priority_map):
        """
        Loop through pixels in the mask and queue the boundary pixels, then loop through the
        created queue and check each for deletability. If so, delete and add its neighbouring pixels
        to the queue as they are now boundary pixels.
        """
        height, width = self.mask.shape
        queue = []
        queue_map = np.zeros_like(self.mask, dtype=bool) # Boolean map of if the pixel is in queue

        # Find all potential pixels to delete
        for row in range(1, height-1):
            for col in range(1, width-1):
                if self.mask[row, col] == 1:
                    # If a 1 touches a 0 it is a boundary pixel
                    if np.min(self.mask[row-1:row+2, col-1:col+2]) == 0:
                        heapq.heappush(queue, (priority_map[row, col], row, col))
                        queue_map[row, col] = True

        while queue:
            _, row, col = heapq.heappop(queue)
            queue_map[row, col] = False
            # Skip if it's been removed from the mask
            if self.mask[row, col] == 0:
                continue

            if self._is_safe_to_delete(row, col):
                self.mask[row, col] = 0
                # Add neighbours in remaining mask to queue as they have become boundaries
                for dirrow, dircol in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    newrow, newcol = row + dirrow, col + dircol
                    if self.mask[newrow, newcol] == 1 and not queue_map[newrow, newcol]:
                        heapq.heappush(queue, (priority_map[newrow, newcol], newrow, newcol))
                        queue_map[newrow, newcol] = True
            else: # Skip if pixel is not safe to delete
                pass


    def _is_safe_to_delete(self, row, col) -> bool:
        """
        Checks if a pixel can be safely deleted. This is determined by checking
        neighoburing pixels and confirming the pixel is not at the end (only 1 neighbour)
        or that it isn't.

        Returns
        -------
        bool
            If the pixel is determined to be on the edge of a 'blob' and will not shrink the
            structure of the skeleton by deleting return true, else return false.
        """
        p = self.get_local_pixels_binary(self.mask, row, col)
        neighbours = [p[1], p[2], p[4], p[7], p[6], p[5], p[3], p[0]]

        # Check that the pixel is not at the end of a line or an isolated dot (num_neighbours < 2)
        # and that the pixel is not surrounded (num_neighbours > 6)
        num_neighbours = sum(neighbours)
        if num_neighbours < 2 or num_neighbours > 6:
            return False

        # Check the pixel's neighbours in a circle, every time a neighbour is 0 and the next one is
        # 1 count that as a transition. A central pixel on the edge of a block will always have one
        # single transition.
        transitions = 0
        for i in range(len(neighbours)):
            if neighbours[i] == 0 and neighbours[(i+1) % 8] == 1:
                transitions += 1

        return transitions == 1


    @staticmethod
    def get_local_pixels_binary(binary_map: npt.NDArray, x: int, y: int) -> npt.NDArray:
        """
        Value of pixels in the local 8-connectivity area around the coordinate (P1) described by x and y.

        P1 must not lie on the edge of the binary map.

        [[p7, p8, p9],    [[0,1,2],
         [p6, P1, p2], ->  [3,4,5], -> [0,1,2,3,5,6,7,8]
         [p5, p4, p3]]     [6,7,8]]

        delete P1 to only get local area.

        Parameters
        ----------
        binary_map : npt.NDArray
            Binary mask of image.
        x : int
            X coordinate within the binary map.
        y : int
            Y coordinate within the binary map.

        Returns
        -------
        npt.NDArray
            Flattened 8-long array describing the values in the binary map around the x,y point.
        """
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        return np.delete(local_pixels, 4)


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
    """
    Create a 2D grid of normalised spatial frequencies for an image.
    Calculate the distance of each pixel from the zero frequency component
    in the Fourier domain.

    Parameters
    ----------
    image :np.ndarray
        2D image to be analysed.

    Returns
    -------
    np.ndarray
        A 2D arrya of the same shape as the input image containing the radial
        normalised frequencies of sqrt(fx^2 + fy^2).
    """
    # Create frequency mask grid
    yres, xres = image.shape
    xr = np.arange(xres)
    yr = np.arange(yres)
    fx = 2 * np.fmin(xr, xres - xr) / xres
    fy = 2 * np.fmin(yr, yres - yr) / yres

    # Full coordinate arrays
    xx, yy = np.meshgrid(fx, fy)
    return np.sqrt(xx**2 + yy**2)

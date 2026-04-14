from __future__ import annotations
import heapq

from loguru import logger
from scipy.ndimage import distance_transform_edt
import skimage as ski
import numpy as np
from cellpose import models, core
from cellpose import utils as cellposeutils
import cv2

from .core.classes import ImageData
from .core.image_processing import get_local_pixels_binary
from .pruning import prune_mask


def segment_image(config: dict[str, any], image_object: ImageData) -> None:
    """
    Start the grain segmentation process by running the method selected in
    config.
    """
    segmentation_method = config["segmentation"]["segmentation_type"]
    if segmentation_method == "traditional":
        segment_image_traditional(config, image_object)
        # Traditional segmentation includes indents and offshoots, which we want to find
        # separately later so these are pruned for now.
        prune_mask(config, image_object)
    elif segmentation_method == "cellpose":
        segment_image_cellpose(config, image_object)


def segment_image_cellpose(config: dict[str, any], image_object: ImageData) -> None:
    """
    Method for using the cellpose ML model to draw outlines of the grains in an image
    and save as a skeletonised mask.

    Parameters
    ----------
    config : dict[str, any]
        A dictionary of config options inputted at the start of the program.
    image_object : ImageData
        Dataclass reference contianing data and stats on the image currently
        being processed.
    """
    logger.info(f"[{image_object.filename}] : *** Mask creation ***")
    logger.info(f"[{image_object.filename}] : Creating grain mask")

    # Skeletonisation config option
    height_bias = config["segmentation"]["height_bias"]

    # If a GPU is available on the device use it, if not then the CPU will be used instead (much slower)
    use_gpu = core.use_gpu()
    if not use_gpu:
        logger.warning(f"[{image_object.filename}] : No Nvidia GPU detected, CPU is being used for segmentation. This may take a few minutes..")

    model = models.CellposeModel(gpu=use_gpu)

    # Parameters:
    # diameter: the px diameter of a grain. As grain sizes differ aim for an average value
    # flow_threshold: How sensitive the segmentation should be. Higher values create more grains, lower values reduce
    # cellprob_threshold: Threshold to mark area as grain based on the probability it is one.
    masks, flows, styles = model.eval(
        image_object.high_pass,
        diameter=50,
        flow_threshold=0.9,
        cellprob_threshold=-1,
        resample=True,
        min_size=15,
    )

    logger.success(f"[{image_object.filename}] : Mask created, Returning image to original size.")
    # Return mask to original shape after model resizing
    if masks.shape != image_object.high_pass.shape:
        masks = cv2.resize(masks.astype(np.float32),
                        (image_object.high_pass.shape[1], image_object.high_pass.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(np.uint16)

    outlines = cellposeutils.masks_to_outlines(masks)
    np_mask = outlines.astype(np.uint16)
    # thick_mask = ski.morphology.dilation(np_mask, ski.morphology.disk(1))
    np_mask = Skeletonisation(image_object.high_pass, np_mask, height_bias=height_bias).do_skeletonisation()

    image_object.mask = np_mask


def segment_image_traditional(config: dict[str, any], image_object: ImageData) -> None:
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

    if image_object.high_pass is not None:
        # For each image create and save a mask
        im = image_object.high_pass
        pixel_to_nm_scaling = image_object.pixel_to_nm_scaling
        height_bias = config["segmentation"]["height_bias"]

        seg_config = config["segmentation"]["traditional"]

        # Scale threshold block size with image scaling and round to nearest odd integer
        threshold_block_size = seg_config["threshold_block_size"] / pixel_to_nm_scaling
        threshold_block_size = 2 * round((threshold_block_size - 1) / 2) + 1

        threshold_offset = seg_config["threshold_offset"]

        # Cleaning config options - adjusted for pixel to nm scaling
        area_threshold = seg_config["cleaning"]["area_threshold"]
        if area_threshold:
            area_threshold = area_threshold / (pixel_to_nm_scaling**2)
            disk_radius = seg_config["cleaning"]["disk_radius_factor"] / pixel_to_nm_scaling
        else:
            disk_radius = None

        # Smoothing config options - adjusted for pixel to nm scaling
        smooth_sigma = seg_config["smoothing"]["sigma"]
        if smooth_sigma:
            smooth_sigma = smooth_sigma / pixel_to_nm_scaling

        logger.info(f"[{image_object.filename}] : *** Mask creation ***")
        logger.info(f"[{image_object.filename}] : Creating grain mask")
        # Create the grain mask by thresholding the image and skeletonising the result
        np_mask = create_grain_mask(
            im,
            threshold_block_size=threshold_block_size,
            threshold_offset=threshold_offset,
            area_threshold=area_threshold,
            disk_radius=disk_radius,
            height_bias=height_bias
        )

        image_object.mask = np_mask


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
        ski.morphology.remove_small_objects(mask.astype(int), max_size=area_threshold)
    )
    return ski.morphology.opening(mask, ski.morphology.disk(disk_radius))


def create_grain_mask(
    im: np.ndarray,
    threshold_block_size: float,
    threshold_offset: float,
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
    # Remove extremes
    perc_low, perc_high = np.percentile(im, [3, 97])
    im = np.clip(im, perc_low, perc_high)

    im_ = ski.filters.difference_of_gaussians(im, low_sigma=1, high_sigma=3)
    # im_ = ski.filters.gaussian(im, sigma=smooth_sigma)
    # Get an array of thresholds for each pixel
    threshold = ski.filters.threshold_local(im_, block_size=threshold_block_size, offset=threshold_offset)
    # threshold = ski.filters.threshold_otsu(im_)
    mask = im_ > threshold
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
        # Extend the image/ mask by mirroring to avoid edge effects
        self.image = np.pad(image, pad_width=1, mode='edge')
        self.mask = np.pad(mask, pad_width=1, mode='edge')
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

        # Remove padding added in __init__() to handle edge pixels
        self.iamge = self.image[1:-1, 1:-1]
        self.mask = self.mask[1:-1, 1:-1]

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

        # Normalise the heightmap
        img_min, img_max = self.image.min(), self.image.max()
        norm_height = (self.image - img_min) / (img_max - img_min + 1e-8)

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
        height, width = self.mask.shape
        if row <= 0 or row >= height - 1 or col <= 0 or col >= width - 1:
            return False

        p = get_local_pixels_binary(self.mask, row, col)
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

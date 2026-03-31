import numpy as np
from scipy import ndimage as ndi
import cv2


def normalise_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalise an array of any size and shape to 0-1. Outliers are also
    ignored (lowest and highest 0.05%).

    Parameters
    ----------
    arr : np.ndarray
        The array (of any shape) the be normalised.

    Returns
    -------
    np.ndarray
        Normalised array with the same shape as the input array.
    """
    # Ignore outlier extremes
    v_min, v_max = np.percentile(arr, [0.05, 99.95])

    clipped = np.clip(arr, v_min, v_max)
    normalised = (clipped - v_min) / (v_max - v_min)
    return normalised


def get_horizontal_gradients(
        image: np.ndarray,
        threshold: float
    ) -> np.ndarray:
    """
    Use the sobel formula to assign a gradient to each pixel in an array
    for the horizontal axis.
    ndi.sobel() is an edge detection formula. `axis=1` means that it tracks horizontal edges
    which is a characteristic of smear areas.

    Parameters
    ----------
    image : np.ndarray
        The image to analyse.
    threshold : float
        The minimum gradient of a pixel to add it to the mask.

    Returns
    -------
    np.ndarray
        Binary mask of all pixels over the horizontal gradient threshold.
    """
    grad_x = ndi.sobel(image, axis=1)
    mask = grad_x > threshold

    return mask


def calculate_rms(image: np.ndarray) -> float:
    """
    Find the RMS (root mean square) of an array.
    This dictates how rough a surface/ image is.

    Parameters
    ----------
    image : np.ndarray
        The 2d image to be analysed

    Returns
    -------
    float
        The RMS value of the inputted array.
    """
    return np.sqrt(np.mean(image**2))


def extend_image(
    image: np.ndarray,
    method: int = cv2.BORDER_REFLECT,
) -> tuple[np.ndarray, dict]:
    """
    Extend image on all sides by specified method.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be extended.
    method : int, optional
        Border type as specified in cv2.

    Returns
    -------
    tuple
        The extended image and a dictionary specifying the size of the borders.

    Raises
    ------
    NotImplementedError
        If `method` is not `cv2.BORDER_REFLECT`.
    """
    if method != cv2.BORDER_REFLECT:
        msg = f"Method {method} not implemented"
        raise NotImplementedError(msg)

    rows, cols = image.shape
    v_ext = rows // 2
    h_ext = cols // 2
    extent = {"top": v_ext, "bottom": v_ext, "left": h_ext, "right": h_ext}

    # Extend the image by mirroring to avoid edge effects
    extended_image = cv2.copyMakeBorder(
        image,
        **extent,
        borderType=method,
    )

    return extended_image, extent


def get_local_pixels_binary(binary_map: np.ndarray, x: int, y: int) -> np.ndarray:
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

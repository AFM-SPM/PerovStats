import numpy as np
from scipy import ndimage as ndi
import cv2


def normalise_array(arr: np.ndarray) -> np.ndarray:
    """Normalise an array of any size and shape to 0-1."""
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
    """
    grad_x = ndi.sobel(image, axis=1)
    mask = grad_x > threshold

    return mask


def calculate_rms(image: np.ndarray) -> float:
    """Find the RMS of an array."""
    return np.sqrt(np.mean(image**2))


def extend_image(
    image: np.ndarray,
    method: int = cv2.BORDER_REFLECT,
) -> tuple[np.ndarray, dict]:
    """
    Extend image by specified method.

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

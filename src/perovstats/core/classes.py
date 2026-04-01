from dataclasses import dataclass

import numpy as np


@dataclass
class Grain:
    """
    Dataclass for storing data on individual grains in an image.

    Parameters
    ----------
    grain_id : int | None
        Unique identifier for the grain.
    grain_mask : np.ndarray | None
        A binary mask of the outline of the singular grain.
    grain_area : float | None
        Area of the grain in nm^2.
    grain_circularity_rating : float | None
        A rating of 0-1 giving how close the shape of the grain is to a circle.
    grain_volume: float | None
        Volume of the grain in nm^3
    """
    grain_id: int
    grain_mask: np.ndarray | None = None
    grain_area: float | None = None
    grain_circularity_rating: float | None = None
    grain_volume: float | None = None

    def to_dict(self) -> dict:
        """
        Create and return a dictionary containing all values to be saved to a csv
        from this class.
        This does not include larger data structures such as np.ndarrays (masks, images).

        Returns
        -------
        dict
            Dictionary of all values to be saved to csv.
        """
        return {
            "grain_id": self.grain_id,
            "grain_area": self.grain_area,
            "grain_circularity": self.grain_circularity_rating,
            "grain_volume": self.grain_volume
        }


@dataclass
class ImageData:
    """
    Dataclass for storing overall data for a processed image,
    including average stats for the grains within the image.

    Parameters
    ----------
    success : bool
        Boolean value, True if all steps completed so far have been successful.
        If False, next steps are not run as data may be missing.
    mask : np.ndarray
        Boolean mask showing grain outlines.
    high_pass : np.ndarray
        Image showing the frequencies left above the frequency cutoff after performing a fourier transform.
    low_pass : np.ndarray
        Image showing the frequencies cut off by a fourier transform.
    smears : np.ndarray
        Mask of smear areas found.
    smears_removed : bool
        True or False depending if smears was run and completed.
    smear_percent : float
        The percentage of the total grain area that was removed by smear detection.
    grains : dict[int, Grain]
        Dictionary containing all grains as class objects, with an id int as the key.
    edge_grains : dict[int, Grain]
        Dictionary containing all grains that touch the edge of the image as class objects, with an id int as the key.
    smear_grains : dict[int, Grain]
        Dictionary containing all grains that touch a smear area as class objects, with an id int as the key.
    file_directory : str
        The folder to save output data to
    filename : str
        The name of the original .spm file without the extension
    mask_rgb : np.ndarray
        Image of the mask with grains coloured in for easier viewing.
    grains_per_nm2 : float
        The amount of grains in the image for every nm^2.
    mask_size_x_nm : float
        The width of the image in nm.
    mask_size_y_nm : float
        The height of the image in nm.
    mask_area_nm : float
        The area of the image in nm^2.
    num_grains : int
        The total number of grains in the image.
    cutoff_freq_nm : float
        The frequency to cutoff during the highpass of the fourier transform in nm.
    cutoff : float
        The actual cutoff for the fourier transform.
    mean_grain_area : float
        The mean area of grains in the image.
    median_grain_area : float
        The median area of grains in the image.
    mode_grain_area : float
        The mode area of grains in the image.
    pixel_to_nm_scaling : float
        Image scaling of pixels to nm.
    threshold : float
        The threshold used for segmentation.
    mask_areas:
        A list of floats containing the areas for each grain.
    circularity_rating:
        A lsit of floats containing the circularity rating for each grain (0-1).
    """
    success: bool | None = None
    image_original: np.ndarray | None = None
    image_flattened: np.ndarray | None = None
    mask: np.ndarray | None = None
    high_pass: np.ndarray | None = None
    low_pass: np.ndarray | None = None
    smears: np.ndarray | None = None
    smears_removed: bool | None = None
    smear_percent: float | None = None
    grains: dict[int, Grain] | None = None
    edge_grains: dict[int, Grain] | None = None
    smear_grains: dict[int, Grain] | None = None
    file_directory: str | None = None
    filename: str | None = None
    mask_rgb: np.ndarray | None = None
    grains_per_nm2: float | None = None
    mask_size_x_nm: float | None = None
    mask_size_y_nm: float | None = None
    mask_area_nm: float | None = None
    num_grains: int | None = None
    cutoff_freq_nm: float | None = None
    cutoff: float | None = None
    mean_grain_area: float | None = None
    median_grain_area: float | None = None
    mode_grain_area: float | None = None
    pixel_to_nm_scaling: float | None = None
    threshold: float | None = None
    mask_areas: list | None = None
    circularity_data: list | None = None

    def to_dict(self) -> dict:
        """
        Create and return a dictionary containing all values to be saved to a csv
        from this class.
        This does not include larger data structures such as np.ndarrays (masks, images).

        Returns
        -------
        dict
            Dictionary of all values to be saved to csv.
        """
        return {
            "file_dir": self.file_directory,
            "filename": self.filename,
            "num_grains": self.num_grains,
            "grains_per_nm2": self.grains_per_nm2,
            "mask_size_x_nm": self.mask_size_x_nm,
            "mask_size_y_nm": self.mask_size_y_nm,
            "mask_area_nm": self.mask_area_nm,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "mean_grain_area_nm2": self.mean_grain_area,
            "median_grain_area_nm2": self.median_grain_area,
            "mode_grain_area_nm2": self.mode_grain_area,
            "smears_removed": self.smears_removed,
            "smear_percent": self.smear_percent,
        }


@dataclass
class PerovStats:
    """
    Class for all data collected in a run of PerovStats consisting of
    the input config for access during the process and data on each image processed.

    Parameters
    ----------
    images : list[ImageData] | None
        A list of all images inputted, containing the class object for each.
    config : dict[str, any] | None
        A dictionary containing all the confg options used, with the key being the name.
    """
    images: list[ImageData] | None = None
    config: dict[str, any] | None = None

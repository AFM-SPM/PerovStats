from dataclasses import dataclass

import numpy as np


@dataclass
class Grain:
    """
    Class for storing individual grains.

    Parameters
    ----------
    grain_id : int | None
        Unique identifier for the grain.
    grain_area : float | None
        Area of the grain in nm^2.
    grain_circularity_rating : float | None
        Unifinished var.
    """
    grain_id: int
    grain_area: float | None = None
    grain_circularity_rating: float | None = None

    def to_dict(self) -> dict:
        return {
            "grain_id": self.grain_id,
            "grain_area": self.grain_area,
            "grain_circularity": self.grain_circularity_rating,
        }


@dataclass
class ImageData:
    """
    Class for storing overall data for a processed image.

    Parameters
    ----------
    topostats_object : any
        TopoStats object generated and used by the filtering stage.
    mask : np.ndarray
        Boolean mask showing grain outlines.
    high_pass : np.ndarray
        Image showing the frequencies left above the frequency cutoff after performing a fourier transform.
    low_pass : np.ndarray
        Image showing the frequencies cut off by a fourier transform.
    grains : dict[int, Grain]
        Dictionary containing all grains as class objects, with an id int as the key.
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
    pixel_to_nm_scaling : float

    """
    image_original: np.ndarray | None = None
    image_flattened: np.ndarray | None = None
    mask: np.ndarray | None = None
    high_pass: np.ndarray | None = None
    low_pass: np.ndarray | None = None
    grains: dict[int, Grain] | None = None
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
    mean_grain_size: float | None = None
    median_grain_size: float | None = None
    mode_grain_size: float | None = None
    pixel_to_nm_scaling: float | None = None

    def to_dict(self) -> dict:
        return {
            "file_dir": self.file_directory,
            "filename": self.filename,
            "num_grains": self.num_grains,
            "grains_per_nm2": self.grains_per_nm2,
            "mask_size_x_nm": self.mask_size_x_nm,
            "mask_size_y_nm": self.mask_size_y_nm,
            "mask_area_nm": self.mask_area_nm,
            "pixel_to_nm_scaling": self.pixel_to_nm_scaling,
            "cutoff_freq_nm": self.cutoff_freq_nm,
            "cutoff": self.cutoff,
            "mean_grain_size_nm2": self.mean_grain_size,
            "median_grain_size_nm2": self.median_grain_size,
            "mode_grain_size_nm2": self.mode_grain_size,
        }


@dataclass
class PerovStats:
    """
    Class for all data collected in a run of PerovStats.

    Parameters
    ----------
    images : list[ImageData] | None
        A list of all images inputted, containing the class object for each.
    config : dict[str, any] | None
        A dictionary containing all the confg options used, with the key being the name.
    """
    images: list[ImageData] | None = None
    config: dict[str, any] | None = None

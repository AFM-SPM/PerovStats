from dataclasses import dataclass
import numpy as np


@dataclass
class Grain:
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
    topostats_object: any
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

    def to_dict(self) -> dict:
        return {
            "file_dir": self.file_directory,
            "filename": self.filename,
            "num_grains": self.num_grains,
            "grains_per_nm2": self.grains_per_nm2,
            "mask_size_x_nm": self.mask_size_x_nm,
            "mask_size_y_nm": self.mask_size_y_nm,
            "mask_area_nm": self.mask_area_nm,
            "cutoff_freq_nm": self.cutoff_freq_nm,
            "cutoff": self.cutoff,
        }


@dataclass
class PerovStats:
    images: list[ImageData] | None = None
    config: dict[str, any] | None = None

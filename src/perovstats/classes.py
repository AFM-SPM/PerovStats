from dataclasses import dataclass
import numpy as np


@dataclass
class Mask:
    mask: np.ndarray
    filename: str
    file_directory: str
    config: dict[str, any]


@dataclass
class Grain:
    area: float | None = None
    circularity_rating: float | None = None


@dataclass
class Grains:
    all_grains: dict[int, Grain]
    filename: str | None = None
    grains_per_nm2: float | None = None
    mask_size_x_nm: float | None = None
    mask_size_y_nm: float | None = None
    mask_area_nm: float | None = None
    num_grains: int | None = None
    dir: str | None = None
    cutoff_freq_nm: float | None = None
    cutoff: float | None = None


@dataclass
class PerovStats:
    filename: str
    mask: Mask | None = None
    grains: Grains | None = None

import numpy as np

class Mask:
    def __init__(
        self,
        mask: np.ndarray,
        filename: str,
        file_directory: str,
        config: dict[str, any]
    ):
        self.mask = mask
        self.filename = filename
        self.file_directory = file_directory
        self.config = config

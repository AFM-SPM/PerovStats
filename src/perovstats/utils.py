import numpy as np

def normalise_array(arr: np.ndarray) -> np.ndarray:
    """Normalise an array of any size and shape to 0-1."""
    v_min, v_max = np.percentile(arr, [0.05, 99.95])

    clipped = np.clip(arr, v_min, v_max)
    normalised = (clipped - v_min) / (v_max - v_min)
    return normalised

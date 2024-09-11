import numpy as np


def scan_horizontally(cell_image: np.ndarray) -> np.ndarray:
    vertical_means = np.mean(cell_image, axis=0)
    # TODO
    return vertical_means



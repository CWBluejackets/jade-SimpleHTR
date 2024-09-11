import numpy as np


def measure_blankness(cell_image: np.ndarray) -> int:
    """
    Find the minimum (darkest) pixel in each column and calculate the
    difference between the lightest and darkest minima across the cell.
    """
    vertical_minima = np.min(cell_image, axis=0)
    min_range = vertical_minima.max() - vertical_minima.min()
    blankness = round(255 - min_range)

# TODO configure the threshold and combine with confidence score


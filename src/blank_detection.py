import numpy as np

from geometry import rolling_average


def measure_blankness(cell_image: np.ndarray, config: dict) -> int:
    """
    Find the minimum (darkest) pixel in each column, get
    the rolling average, and calculate the lightest one and
    the median (paper) of the cell.
    @param cell_image: the cell
    @param config: general configuration
    """
    window = config['blank_detection']['smoothing_window']
    # value of darkest pixel in each column
    vertical_minima = np.min(cell_image, axis=0)
    # rolling average
    vertical_minima = rolling_average(vertical_minima, window)
    # cell median is value of the paper
    # (using vertical_minima.max() breaks when the cell is full of handwriting)
    blankness = round(255 - np.median(cell_image) + vertical_minima.min())
    return blankness

# TODO configure the threshold and combine with confidence score


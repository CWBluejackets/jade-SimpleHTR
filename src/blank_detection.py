import numpy as np


def measure_blankness(cell_image: np.ndarray, config: dict) -> int:
    """
    Find the minimum (darkest) pixel in each column, get
    the rolling average, and calculate the lightest one and
    the median (paper) of the cell.
    @param cell_image: the cell
    @param config: general configuration
    @return: higher number is more blank
    """
    window = config['blank_detection']['smoothing_window']
    # value of darkest pixel in each column
    vertical_minima = np.min(cell_image, axis=0)
    # rolling average
    vertical_minima = roll(vertical_minima, window)
    # The median is the value of the paper.
    # (Using vertical_minima.max() breaks when the cell width is full of handwriting.)
    blankness = round(255 - np.median(cell_image) + vertical_minima.min())
    return blankness


def roll(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the rolling average of a series.
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    :param values: list of values to average:
    :param window_size: size:
    :return: array of length original - window + 1 (without boundary effects)
    """
    if window_size < 2:
        result = values
    else:
        slider = np.ones(window_size) / window_size
        # mode='same' creates boundary effects that mess up the minimum
        result = np.convolve(values, slider, 'valid')
    return result

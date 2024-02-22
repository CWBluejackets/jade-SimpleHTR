# library file

import math
from collections import Counter
from typing import Tuple

import cv2
import numpy as np


# point = (x, y)
# in cv2, y = 0 at the top
#         x = 0 at the left
# cv2 ndarray.shape = (height, width)

# NOTE: cv2.line(...) correctly handles endpoints outside the image without
# raising exceptions, so extrapolation does NOT need boundary safety

def solve_line(point0, point1) -> Tuple[float, float]:
    """
    Solve y = ax + b from two points
    :param point0: (x, y)
    :param point1: (x, y)
    :return: a, b
    """
    a = (point1[1] - point0[1]) / (point1[0] - point0[0])
    b = point1[1] - a * point1[0]
    return a, b


def extrapolate_vertical(point0, point1, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Extrapolate to the top and bottom edges.
    :param point0:
    :param point1:
    :param image:
    :return: top_point, bottom_point, horizontal_difference
    """
    max_x, max_y = get_maxima(image)
    x0, y0 = point0
    x1, y1 = point1

    # special case: vertical line has infinite slope
    if x0 == x1:
        return (x0, 0), (x0, max_y), 0

    a, b = solve_line(point0, point1)
    top = (round(-b / a), 0)
    bottom = (round((max_y - b) / a), max_y)
    x_diff = abs(bottom[0] - top[0])
    return top, bottom, x_diff


def extrapolate_horizontal(point0, point1, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Extrapolate to the left and right edges.
    :param point0:
    :param point1:
    :param image:
    :return: left_point, right_point, vertical_difference
    """
    max_x, max_y = get_maxima(image)
    a, b = solve_line(point0, point1)

    left = (0, round(b))
    right = (max_x, round(a * max_x + b))
    y_diff = abs(right[1] - left[1])

    return left, right, y_diff


def measure(line0):
    """
    Measure the length and direction of a line (CW from +x axis).
    Note: because in opencv y increases down the image, the angles
    are backwards from usual trigonometric convention.
    :param line0: (x0, y0, x1, y1) format returned by HoughLinesP
    :return: degrees, length, line (repeated in same format)
    """
    dy = line0[3] - line0[1]
    dx = line0[2] - line0[0]
    angle = math.degrees(math.atan2(dy, dx))
    # Pythagoras
    length = math.sqrt(dy * dy + dx * dx)
    return angle, length, line0


def normalize_angle(angle):
    """
    Normalize the angle to the range -45° <= angle < 135° to simplify
    horizontal and vertical detection.
    :param angle:
    :return:
    """
    while angle < -45:
        angle += 180
    while angle >= 135:
        angle -= 180
    return angle


def is_horizontal(angle, tolerance):
    norm_angle = normalize_angle(angle)
    return (0 - tolerance) < norm_angle < (0 + tolerance)


def is_vertical(angle, tolerance):
    norm_angle = normalize_angle(angle)
    return (90 - tolerance) < norm_angle < (90 + tolerance)


def get_maxima(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[0] - 1, image.shape[1] - 1


def binned_mode(values, bins=10, range=None) -> float:
    """
    Make histogram bins and find the mode.
    :param values:
    :param bins: number of bins
    :param range: (otherwise determined from values)
    :return: the mode
    """
    # TODO maybe implement the proper histogram mode formula
    # https://www.statology.org/histogram-mode/
    if not range:
        range0 = (values.min(), values.max())
    else:
        range0 = range
    counts, lefts = np.histogram(values, bins=bins, range=range0)
    i = counts.argmax()
    return (lefts[i] + lefts[i + 1]) / 2.0


def most_frequent_values(values, interval, end, count):
    bins = range(0, end, interval)
    counts, edges = np.histogram(values, bins=bins)
    middles = [round((edges[i] + edges[i+1])/2) for i in range(len(edges) - 1)]
    d = {middles[i]: counts[i] for i in range(len(counts)) if (counts[i] > 0)}
    c = Counter(d)
    return c.most_common(count)


def identify_image_sides(overall_median, left_median, right_median):
    """
    Identify left and right ends of image.
    :return:
    """
    left_result = median_result(left_median, overall_median)
    right_result = median_result(right_median, overall_median)
    return left_result, right_result


def get_medians(image: np.ndarray, config):
    """
    Get medians for the whole image and the left and right parts.
    Note that the array has shape (vertical, horizontal)
    :param image: greyscale image
    :param config: config file object
    :return: overall, left, right
    """
    overall_median = np.median(image)
    left = round(image.shape[1] * config['edge_test']['horizontal_band'])
    top = round(image.shape[0] * config['edge_test']['vertical_exclusion'])
    left_median = np.median(image[top:-top, :left])
    right_median = np.median(image[top:-top, -left:])
    return overall_median, left_median, right_median


def median_result(test_median, overall_median):
    if test_median >= (overall_median + 3):
        result = 'light'
    elif test_median < (overall_median / 2):
        result = 'dark'
    else:
        result = 'unknown'
    return result


def window_average(values, window_size):
    """
    Calculate the window average over a series of values
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    :param values: list of values to average:
    :param window_size: size:
    :return: array with the same size as the input array
    """
    return np.convolve(values, np.ones(window_size), 'same') / window_size


def vertical_medians(image: np.ndarray):
    """
    Calculate the median of each vertical line in the image
    :param image: greyscale image
    :return: array of medians with the image width
    """
    return np.median(image, axis=0)


def smoothed_vertical_medians(image: np.ndarray, config):
    """
    Calculate the smoothed median of each vertical line
    :param image: greyscale image
    :param config: config file object
    :return: array with the image width
    """
    medians = vertical_medians(image)
    window = config['edge_detection']['window_size']
    return window_average(medians, window)


def erase_margins(image: np.ndarray, config) -> np.ndarray:
    """
    Determine the left and right cutoffs off the main form and overwrite
    the edges with the image median.
    :param image: greyscale image
    :param config: config file object
    :return:
    """
    overall_median, left_median, right_median = get_medians(image, config)
    edge_types = identify_image_sides(overall_median, left_median, right_median)
    v_medians = smoothed_vertical_medians(image, config)
    x_lh = get_left_cutoff(v_medians, edge_types[0])
    x_rh = get_right_cutoff(v_medians, edge_types[1])
    y_max = image.shape[0] - 1
    modified_image = image.copy()
    for x in range(0, x_lh + 1):
        cv2.line(modified_image, (x, 0), (x, y_max), overall_median, 1)
    for x in range(x_rh, image.shape[1]):
        cv2.line(modified_image, (x, 0), (x, y_max), overall_median, 1)
    return modified_image


def get_left_cutoff(medians, edge_type, overall_median):
    max_left = len(medians) // 4
    cutoff = 0
    if edge_type == 'light':
        for x in range(max_left, 0, -1):
            if medians[x] - overall_median > 5:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(max_left, 0, -1):
            if medians[x] < overall_median * 0.8:
                cutoff = x
                break
    return cutoff


def get_right_cutoff(medians, edge_type, overall_median):
    min_right = len(medians) * 3 // 4
    cutoff = len(medians) - 1
    if edge_type == 'light':
        for x in range(min_right, len(medians)):
            if medians[x] - overall_median > 5:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(min_right, len(medians)):
            if medians[x] < overall_median * 0.8:
                cutoff = x
                break
    return cutoff





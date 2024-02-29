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

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)


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
        range0 = (min(values), max(values))
    else:
        range0 = range
    counts, lefts = np.histogram(values, bins=bins, range=range0)
    i = counts.argmax()
    return (lefts[i] + lefts[i + 1]) / 2.0


def most_frequent_values(values, interval, end, count):
    """
    Bin the values histogram-style and return the middles and
    counts of the most frequent bins
    :param values: sequence of values
    :param interval: bin size
    :param end: high value of highest bin
    :param count: number of values to return
    :return:
    """
    bins = range(0, end, interval)
    counts, edges = np.histogram(values, bins=bins)
    middles = [round((edges[i] + edges[i+1])/2) for i in range(len(edges) - 1)]
    d = {middles[i]: counts[i] for i in range(len(counts)) if (counts[i] > 0)}
    c = Counter(d)
    return c.most_common(count)


def identify_image_sides(overall_median, left_median, right_median) -> Tuple[str, str]:
    """
    Identify left and right ends of image as 'light', 'dark', or 'unknown'
    :return:
    """
    left_result = median_result(left_median, overall_median)
    right_result = median_result(right_median, overall_median)
    return left_result, right_result


def get_medians(image: np.ndarray, config) -> Tuple[int, int, int]:
    """
    Get medians for the whole image and the left and right parts.
    Note that the array has shape (vertical, horizontal)
    :param image: greyscale image
    :param config: config file object
    :return: overall, left, right
    """
    overall_median = np.median(image)
    left = round(image.shape[1] * config['margin_detection']['left_right_test'])
    left_median = np.median(image[:, :left])
    right_median = np.median(image[:, -left:])
    return overall_median, left_median, right_median


def median_result(test_median, overall_median) -> str:
    """
    Determine the condition of the left and white sides (binding or black background)
    :param test_median:
    :param overall_median:
    :return:
    """
    if test_median >= (overall_median + 3):
        result = 'light'
    elif test_median < (overall_median / 2):
        result = 'dark'
    else:
        result = 'unknown'
    return result


def rolling_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the window average over a series of values
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    :param values: list of values to average:
    :param window_size: size:
    :return: array with the same size as the input array
    """
    return np.convolve(values, np.ones(window_size), 'same') / window_size


def smoothed_vertical_medians(image: np.ndarray, config) -> np.ndarray:
    """
    Calculate the smoothed median of each vertical line
    :param image: greyscale image
    :param config: config file object
    :return: array with the image width
    """
    medians = np.median(image, axis=0)
    window = config['margin_detection']['smoothing_window']
    return rolling_average(medians, window)


def smoothed_horizontal_medians(image: np.ndarray, config):
    """
    Calculate the smoothed median of each horizontal line
    :param image: greyscale image
    :param config: config file object
    :return: array with the image height
    """
    medians = np.median(image, axis=1)
    window = config['margin_detection']['smoothing_window']
    return rolling_average(medians, window)


def erase_margins(image: np.ndarray, config) -> np.ndarray:
    """
    Determine the left and right cutoffs off the main form and overwrite
    the edges with the image median.
    :param image: greyscale image
    :param config: config file object
    :return:
    """
    # set some values
    image_median = np.median(image)
    x_max = image.shape[1] - 1
    y_max = image.shape[0] - 1

    # scan horizontal lines
    top_cutoff, bottom_cutoff = get_top_bottom_cutoffs(image, config)
    modified_image = image.copy()
    # erase the top and bottom margins
    for y in range(0, top_cutoff):
        cv2.line(modified_image, (0, y), (x_max, y), image_median, 1)
    for y in range(bottom_cutoff, y_max):
        cv2.line(modified_image, (0, y), (x_max, y), image_median, 1)

    # scan vertical lines from the MODIFIED image
    overall_median, left_median, right_median = get_medians(modified_image, config)
    side_types = identify_image_sides(overall_median, left_median, right_median)
    print(side_types)
    v_medians = smoothed_vertical_medians(modified_image, config)
    x_lh = get_left_cutoff(v_medians, side_types[0], overall_median, config)
    x_rh = get_right_cutoff(v_medians, side_types[1], overall_median, config)
    # and erase the left and right margins
    for x in range(0, x_lh + 1):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    for x in range(x_rh, image.shape[1]):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    return modified_image


def get_top_bottom_cutoffs(image: np.ndarray, config) -> Tuple[int, int]:
    """
    Scan horizontal lines to find the top and bottom margins
    :param image: greyscale image
    :param config: config file object
    :return: two y-values
    """
    y_max = image.shape[0] - 1
    image_median = np.median(image)
    black_limit = image_median * config['margin_detection']['black_limit']
    y_fraction = config['margin_detection']['top_bottom_limit']
    h_medians = smoothed_horizontal_medians(image, config)
    top_cut = 0
    for y in range(0, round(image.shape[0] * y_fraction)):
        if h_medians[y] > black_limit:
            top_cut = y
            break

    bottom_cut = y_max
    for y in range(y_max, round(y_max * (1 - y_fraction)), -1):
        if h_medians[y] > black_limit:
            bottom_cut = y
            break
    return top_cut, bottom_cut


def get_left_cutoff(medians, edge_type, overall_median, config) -> int:
    """
    Find the maximum X value for the left margin
    :param medians: smoothed medians of vertical lines
    :param edge_type: 'light', 'dark', or 'unknown'
    :param overall_median: median of the whole image
    :param config: config file object
    :return:
    """
    black_limit = overall_median * config['margin_detection']['black_limit']
    max_left = round(len(medians) * config['margin_detection']['left_right_limit'])
    cutoff = 0
    if edge_type == 'light':
        h_gradient = np.gradient(medians)
        gradient_cutoff = config['margin_detection']['gradient_cutoff']
        for x in range(max_left, 0, -1):
            if h_gradient[x] > gradient_cutoff:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(0, max_left):
            if medians[x] > black_limit:
                cutoff = x
                break
    return cutoff


def get_right_cutoff(medians, edge_type, overall_median, config) -> int:
    """
    Find the minimum X value for the right margin
    :param medians: smoothed medians of vertical lines
    :param edge_type: 'light', 'dark', or 'unknown'
    :param overall_median: median of the whole image
    :param config: config file object
    :return:
    """
    black_limit = overall_median * config['margin_detection']['black_limit']
    min_right = round(len(medians) * (1 - config['margin_detection']['left_right_limit']))
    cutoff = len(medians) - 1
    if edge_type == 'light':
        h_gradient = np.gradient(medians)
        gradient_cutoff = - config['margin_detection']['gradient_cutoff']
        for x in range(min_right, len(medians)):
            if h_gradient[x] < gradient_cutoff:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(len(medians) - 1, min_right, -1):
            if medians[x] > black_limit:
                cutoff = x
                break
    return cutoff


def generate_test_image(image, horizontal_lines, vertical_lines, debug):
    if not debug:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for measured_line in horizontal_lines + vertical_lines:
        line = measured_line[2]
        cv2.line(image_rgb, (line[0], line[1]), (line[2], line[3]), BLUE, 1)
    return image_rgb




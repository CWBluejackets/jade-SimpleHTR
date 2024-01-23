# library file

import math
from typing import Tuple

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


def binned_mode(values, bins=10, range=None):
    # TODO maybe implement the proper histogram mode formula
    # https://www.statology.org/histogram-mode/
    if not range:
        range0 = (values.min(), values.max())
    else:
        range0 = range
    counts, lefts = np.histogram(values, bins=bins, range=range0)
    i = counts.argmax()
    return (lefts[i] + lefts[i + 1]) / 2.0


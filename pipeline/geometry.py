# library file

import math
from typing import Tuple

import numpy as np


# point = (x, y)
# in cv2, y = 0 = top
#         x = 0 = left
# cv2 ndarray.shape = (height, width)

# NOTE: cv2.line(...) can handle endpoints outside the image correctly and without
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
    Extrapolate to the left and right edges. Warning: nothing yet to guard
    against negative x-values (or past the RH edge)
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
    Measure the length and direction of a line
    :param line0: (x0, y0, x1, y1) format returned by HoughLinesP
    :return: degrees (positive float), length, line (repeated in same format)
    """
    dy = line0[3] - line0[1]
    dx = line0[2] - line0[0]
    # arctangent in radians
    angle_radians = math.atan2(dy, dx)
    # convert to degrees
    angle_degrees = angle_radians * 180 / math.pi
    # normalize to positive angle
    if angle_degrees < 0:
        angle_degrees += 180
    # Pythagoras
    length = math.sqrt(dy * dy + dx * dx)
    return angle_degrees, length, line0


def is_horizontal(angle, tolerance):
    return ((0 - tolerance) < angle < (0 + tolerance)) or ((180 - tolerance) < angle < (180 + tolerance))


def is_vertical(angle, tolerance):
    return ((90 - tolerance) < angle < (90 + tolerance)) or ((270 - tolerance) < angle < (270 + tolerance))


def get_maxima(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[0] - 1, image.shape[1] - 1

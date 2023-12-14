# library file

import math
from typing import Tuple

import numpy as np


# point = (x, y)
# y = 0 = top
# x = 0 = left
# cv2 ndarray.shape = (height, width)


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


def extrapolate_vertical(point0, point1, image: np.ndarray) -> tuple:
    """
    Extrapolate to the top and bottom edges. Warning: nothing yet to guard
    against negative x-values (or past the bottom edge)
    :param point0:
    :param point1:
    :param image:
    :return: top_point, bottom_point, horizontal_difference
    """
    max_x, max_y = get_maxima(image)
    x0, y0 = point0
    x1, y1 = point1

    # special case: vertical lines has infinite slope
    if x0 == x1:
        return (x0, 0), (x0, max_y), 0

    a, b = solve_line(point0, point1)
    top = (-b / a, 0)
    bottom = ((max_y - b) / a, max_y)

    y_top = 0
    y_bottom = image.shape[0] - 1
    x_upper, y_upper = min(point0, point1, key=lambda p: p[1])
    x_lower, y_lower = max(point0, point1, key=lambda p: p[1])
    x_top = (y_top - y_lower) * (x_upper - x_lower) / (y_upper - y_lower) + x_lower
    x_top = int(x_top)
    x_bottom = (y_bottom - y_upper) * (x_upper - x_lower) / (y_upper - y_lower) + x_upper
    x_bottom = int(x_bottom)
    return (x_top, y_top), (x_bottom, y_bottom), abs(x_top - x_bottom)


def extrapolate_horizontal(point0, point1, image_array):
    """
    Extrapolate to the left and right edges. Warning: nothing yet to guard
    against negative x-values (or past the RH edge)
    :param point0:
    :param point1:
    :param image_array:
    :return: left_point, right_point, vertical_difference
    """
    x_left = 0
    x_right = image_array.shape[1] - 1
    x_0, y_0 = min(point0, point1, key=lambda p: p[0])
    x_1, y_1 = max(point0, point1, key=lambda p: p[0])

    y_left = y_1 - (x_1 - x_left) * (y_1 - y_0) / (x_1 - x_0)
    y_left = int(y_left)
    y_right = (x_right - x_0) * (y_1 - y_0) / (x_1 - x_0) + y_0
    y_right = int(y_right)
    return (x_left, y_left), (x_right, y_right), abs(y_right - y_left)


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


def is_within_image(point, max_x, max_y):
    return (0 <= point[0] <= max_x) and (0 <= point[1] <= max_y)


def get_maxima(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[0] - 1, image.shape[1] - 1

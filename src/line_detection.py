import math
from typing import List

import cv2
import numpy as np

# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga26413521fde978c0c97f11149336f6e1
# QUOTE: The LineSegmentDetector algorithm is defined using the standard values. Only
# advanced users may want to edit those, as to tailor it for their own application.
LSD = cv2.createLineSegmentDetector()


def normalize_angle(angle):
    """
    Normalize the angle to the range -45° <= angle < 135° to simplify
    horizontal and vertical detection.
    :param angle: output of atan2 converted to degrees
    :return: normalized angle
    """
    if angle < -45:
        angle += 180
    elif angle >= 135:
        angle -= 180
    return angle


class Line:
    def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.x0 = round(x0)
        self.y0 = round(y0)
        self.x1 = round(x1)
        self.y1 = round(y1)
        dy = y1 - y0
        dx = x1 - x0
        # result is -180° to +180°
        angle = math.degrees(math.atan2(dy, dx))
        self.angle = normalize_angle(angle)
        self.length = math.hypot(dx, dy)
        return

    def is_nearly_horizontal(self, tolerance):
        return -tolerance <= self.angle <= tolerance

    def is_nearly_vertical(self, tolerance):
        return (90 - tolerance) <= self.angle <= (90 + tolerance)


def detect_lines_within(image: np.ndarray, left: int, right: int, top: int, bottom: int) -> List[Line]:
    """
    Crop the image to the specified coordinates and detect lines within that area.
    :param image: the full image
    :param left: left X of inner image
    :param right: right X of inner image
    :param top: top Y of inner image
    :param bottom: bottom Y of inner image
    :return: list of Line objects
    """
    inner_image = image[top:bottom + 1, left:right + 1]
    lines_and_stuff = LSD.detect(inner_image)
    lines = convert_detection_array(lines_and_stuff[0], x_plus=left, y_plus=top)
    return lines


def convert_detection_array(lines_array: np.ndarray, x_plus=0, y_plus=0) -> list:
    """
    Transform the output of line detection into measured line objectsfor ease of use
    :param lines_array: output of LSD or Hough
    :param x_plus: x coordinate of top left corner of inner array
    :param y_plus: y coordinate of top left corner of inner array
    :return: list of Line objects
    """
    coordinates = [thing[0] for thing in lines_array.tolist()]
    return [Line(c[0] + x_plus, c[1] + y_plus, c[2] + x_plus, c[3] + y_plus) for c in coordinates]


def keep_horizontal(lines: List[Line], tolerance: float) -> List[Line]:
    return [line for line in lines if line.is_nearly_horizontal(tolerance)]


def keep_vertical(lines: List[Line], tolerance: float) -> List[Line]:
    return [line for line in lines if line.is_nearly_vertical(tolerance)]


def keep_horizontal_and_vertical(lines: List[Line], h_tolerance: float,
                                 v_tolerance: float) -> List[Line]:
    return [line for line in lines
            if line.is_nearly_vertical(v_tolerance) or line.is_nearly_horizontal(h_tolerance)]



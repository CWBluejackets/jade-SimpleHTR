# library file
import logging
import math
import os

import cv2
import numpy as np

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)


def process_sheet(filename, config):
    output_directory = config['png_directory']
    logging.info(f'Output to {output_directory}')
    logging.info(f'Reading image from {filename}')
    image = safe_imread(filename, cv2.IMREAD_GRAYSCALE, exception_if_missing=config['img_file_exception'])
    raw_lines = extract_raw_lines(image, config)
    # TODO
    return


def measure(line0):
    """
    Measure the length and direction of a line
    :param line0: (x0, y0, x1, y1)
    :return: degrees (90 up), length, line (same format)
    """
    dy = line0[3] - line0[1]
    dx = line0[2] - line0[0]
    # angle in radians
    angle_radians = math.atan2(dy, dx)
    # angle in degrees
    angle_degrees = angle_radians * 180 / math.pi
    if angle_degrees < 0:
        angle_degrees += 180
    length = math.sqrt(dy * dy + dx * dx)
    return angle_degrees, length, line0


def is_horizontal(angle, tolerance):
    return (- tolerance < angle < + tolerance) or (180 - tolerance < angle < 180 + tolerance)


def extract_raw_lines(image, config):
    dst = cv2.Canny(image, config['canny_min'], config['canny_max'], None,
                    config['canny_aperture'], True)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, config['hough_threshold'],
                            None, config['hough_min_length'], config['hough_max_gap'])
    return lines


def safe_imread(img_path, cv2_flags, exception_if_missing=False):
    if os.path.exists(img_path):
        result = cv2.imread(img_path, cv2_flags)
    else:
        result = None
        message = f'File {img_path} is missing!'
        if exception_if_missing:
            raise FileNotFoundError(message)
        else:
            print(message)
    return result

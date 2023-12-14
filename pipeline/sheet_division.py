# library file

import logging
import os

import cv2
import numpy as np

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)


def process_sheet(filename, config, workflows):
    """
    Process sheet...
    :param filename: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :return:
    """
    output_directory = config['png_directory']
    logging.info(f'Output to {output_directory}')
    logging.info(f'Reading image from {filename}')
    image = safe_imread(filename, cv2.IMREAD_GRAYSCALE, exception_if_missing=config['img_file_exception'])
    raw_lines = extract_raw_lines(image, config)
    # TODO
    return


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

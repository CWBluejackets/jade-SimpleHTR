# library file

import logging

import cv2
import numpy as np

import geometry
import utilities

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)


# workflow specification:
# key int -> { 'description': string, 'numeric': bool }

# format specification:
# key str 'type_<#>'
# -> { 'columns' -> list of column specifications (LTR),
#      'example': str url,
#      'first_year': int, 'form_code': str, 'type_name': str }

# column specification
# { 'index': int, 'name': str,
#   'process': bool, 'skip': bool,
#   'workflow': None or int,
#   'width': None or int or list of int (maybe) }


def process_sheet(filename, output_dir, config, workflows, formats):
    """
    Process sheet...
    :param output_dir: directory for PNGs and control files
    :param formats: column format specifications
    :param filename: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :return:
    """
    output_directory = config['png_directory']
    logging.info(f'Output to {output_directory}')
    logging.info(f'Reading image from {filename}')
    image = utilities.safe_imread(filename, cv2.IMREAD_GRAYSCALE,
                                  exception_if_missing=config['img_file_exception'])
    raw_lines = extract_raw_lines(image, config)
    # HoughLinesP output shape is (nbr_lines, 1, 4) where the 4 is (x0, y0, x1, y1)
    measured_lines = [geometry.measure(line[0]) for line in raw_lines.tolist()]
    vertical_lines = [line for line in measured_lines
                      if geometry.is_vertical(line[0], config['columns']['deviation'])]
    horizontal_lines = [line for line in measured_lines
                        if geometry.is_horizontal(line[0], config['rows']['deviation'])]
    # TODO filter vertical lines
    #      check against formats
    # TODO filter horizontal lines
    #      check against approximate multiples of config['rows']['typical']
    return


def extract_raw_lines(image, config) -> np.ndarray:
    dst = cv2.Canny(image, config['canny_min'], config['canny_max'], None,
                    config['canny_aperture'], True)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, config['hough_threshold'],
                            None, config['hough_min_length'], config['hough_max_gap'])
    return lines



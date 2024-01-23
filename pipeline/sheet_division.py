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


def process_sheet(input_jpg_file, output_dir, config, workflows, formats):
    """
    Process sheet...
    :param output_dir: directory for PNGs and control files
    :param formats: column format specifications
    :param input_jpg_file: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :return:
    """
    logging.info(f'Output to {output_dir}')
    logging.info(f'Reading image from {input_jpg_file}')
    vessel, page = utilities.get_vessel_and_page(input_jpg_file)
    control_data = {'muster': {'vessel': vessel,
                               'page': page}}
    image = utilities.safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                  exception_if_missing=config['img_file_exception'])
    rotation = get_image_rotation(image, config)

    # TODO rotate and proceed
    
    # measured_lines = extract_measured_lines(image, config)
    # vertical_lines = [line for line in measured_lines
    #                   if geometry.is_vertical(line[0], config['columns']['deviation'])]
    # horizontal_lines = [line for line in measured_lines
    #                     if geometry.is_horizontal(line[0], config['rows']['deviation'])]
    # TODO filter vertical lines
    #      check against formats; use to determine most likely format
    # TODO filter horizontal lines
    #      check against format['row_height_top'] and approximate multiples of format['row_height']
    #      and use config['rows']['height_tolerance']
    # TODO detect the rotation of the image using the typical variation from 0 or 90° of nearly H or V lines
    #      and rotate the whole image to rectify that
    # TODO generate boxes and record everything in the control data
    # merge adjacent columns with same workflow ID
    utilities.write_control_file(input_jpg_file, output_dir, control_data)
    return


def extract_raw_lines(image, config) -> np.ndarray:
    dst = cv2.Canny(image, config['canny_min'], config['canny_max'], None,
                    config['canny_aperture'], True)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, config['hough_threshold'],
                            None, config['hough_min_length'], config['hough_max_gap'])
    return lines


def extract_measured_lines(image, config) -> list:
    raw_lines = extract_raw_lines(image, config)
    # HoughLinesP output shape is (nbr_lines, 1, 4) where the 4 is (x0, y0, x1, y1)
    measured_lines = [geometry.measure(line[0]) for line in raw_lines.tolist()]
    return measured_lines


def get_image_rotation(image: np.ndarray, config) -> float:
    """
    Determine the rotation of the nearly horizontal lines.
    :param image: array from cv2
    :param config:
    :return: angle CW in degrees
    """
    all_lines = extract_measured_lines(image, config)
    horizontal_lines = [line for line in all_lines
                        if geometry.is_horizontal(line[0], config['rows']['deviation'])]
    angles = [line(0) for line in horizontal_lines]
    rotation = geometry.binned_mode(angles)
    return rotation



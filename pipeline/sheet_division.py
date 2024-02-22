# library file

import logging
from typing import Tuple

import cv2
import numpy as np

import formatting
import geometry
import utilities
import scipy.ndimage as ndimage


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


def process_muster_sheet(input_jpg_file, image_metadata, output_dir, config, workflows, formats):
    """
    Process sheet...
    :param image_metadata:
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
    muster_date = image_metadata.get('muster_date', 'unknown')
    control_data = {'muster': {'vessel': vessel,
                               'page': page,
                               'date': muster_date}}
    vertical_lines, horizontal_lines = process_image(input_jpg_file, output_dir, config)

    # convert vertical lines to a vector feature for format classification
    format_type, format_info = formatting.get_format(vertical_lines, formats, workflows, image_metadata['muster_date'])
    column_specifications = formatting.get_columns(vertical_lines, formats, workflows, format_type, format_info)

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


def process_image(input_jpg_file, config, debug=False) -> Tuple[list, list, np.ndarray]:
    image_raw = utilities.safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                      exception_if_missing=config['img_file_exception'])
    # determine and correct the rotation
    rotation = get_image_rotation(image_raw, config)
    image = ndimage.rotate(image_raw, angle=rotation, reshape=False)

    # overwrite the left and write junk to eliminate false lines
    image = geometry.erase_margins(image, config)

    # TODO use looser thresholds for line detection now
    lines = extract_measured_lines(image, config)
    vertical_lines = [line for line in lines
                      if geometry.is_vertical(line[0], config['columns']['deviation'])]
    horizontal_lines = [line for line in lines
                        if geometry.is_horizontal(line[0], config['rows']['deviation'])]
    # TODO generate a test image with lines marked
    test_image = geometry.generate_test_image(image, horizontal_lines, vertical_lines, debug)
    return vertical_lines, horizontal_lines, test_image


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
    Determine the rotation angle from the nearly horizontal lines.
    :param image: array from cv2
    :param config:
    :return: angle CW in degrees
    """
    all_lines = extract_measured_lines(image, config)
    horizontal_lines = [line for line in all_lines
                        if geometry.is_horizontal(line[0], config['rows']['initial_deviation'])]
    angles = [line(0) for line in horizontal_lines]
    rotation = geometry.binned_mode(angles)
    return rotation

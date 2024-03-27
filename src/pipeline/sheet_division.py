# library file

import logging

import cv2

import geometry
import utilities


def process_muster_sheet(input_jpg_file, image_metadata, output_dir, config, workflows, templates,
                         exception_if_missing):
    """
    Process sheet...
    :param image_metadata:
    :param output_dir: directory for PNGs and control files
    :param formats: column format specifications
    :param input_jpg_file: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :param exception_if_missing: throw exception for missing file
    :return:
    """
    logging.info(f'Output to {output_dir}')
    logging.info(f'Reading image from {input_jpg_file}')
    vessel, page = utilities.get_vessel_and_page(input_jpg_file)
    muster_date = image_metadata.get('muster_date', 'unknown')
    format_type = image_metadata.get('format_type', 'type2')
    control_data = {'muster': {'vessel': vessel,
                               'page': page,
                               'date': muster_date}}

    image_raw = utilities.safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                      exception_if_missing=exception_if_missing)
    template_image = templates[format_type]['image']

    registered_image = geometry.adjust_image(image_raw, template_image, config)

    # TODO

    utilities.write_control_file(input_jpg_file, output_dir, control_data)
    return








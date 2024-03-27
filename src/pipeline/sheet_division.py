# library file

import logging

import cv2
import numpy as np

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
    muster_date = image_metadata.get('muster_date', 'unknown')
    format_type = image_metadata.get('format_type', '2')

    image_raw = utilities.safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                      exception_if_missing=exception_if_missing)
    template = templates[format_type]
    template_image = template['image']

    registered_image = geometry.adjust_muster_to_template(image_raw, template_image, config)
    for cell in template['cells']:
        wf_id = cell['workflow']
        model = workflows[wf_id]['model']
        threshold = workflows[wf_id]['confidence']
        cell_result = process_cell(registered_image, cell, model, threshold)

    utilities.write_control_file(input_jpg_file, output_dir)
    return


def process_cell(image: np.ndarray, cell, model, threshold: float):
    cell_output = {'threshold': threshold}
    cell_output.update(cell)
    cell_image = image[cell['top']: cell['bottom']+1, cell['left']: cell['right']+1]
    text, confidence = model.apply(cell_image)
    cell_output['text'] = text
    cell_output['confidence'] = confidence
    return cell_output



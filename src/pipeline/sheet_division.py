# library file
import json
import logging
import os.path

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
    :param templates: template specifications
    :param input_jpg_file: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :param exception_if_missing: throw exception for missing file
    :return:
    """
    logging.info(f'Output to {output_dir}')
    logging.info(f'Reading image from {input_jpg_file}')
    format_type = image_metadata.get('format_type', '2')
    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]

    image_raw = utilities.safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                      exception_if_missing=exception_if_missing)
    template = templates[format_type]
    template_image = template['image']

    registered_image = geometry.adjust_muster_to_template(image_raw, template_image, config)
    for sequence, cell in enumerate(template['cells']):
        wf_id = cell['workflow']
        model = workflows[wf_id]['model']
        threshold = workflows[wf_id]['confidence']
        cell_result = process_cell(registered_image, cell, model, threshold)
        cell_result.update(image_metadata)
        output_filename = os.path.join(output_dir,
                                       f'{output_prefix}_{sequence:04d}.json')
        print(f'Writing {output_filename}')
        with open(output_filename, 'w') as f:
            json.dump(cell_result, f)

        # TODO generate image for checking if cell['confidence'] < threshold
    return


def process_cell(image: np.ndarray, cell, model, threshold: float):
    cell_output = {'threshold': threshold}
    cell_output.update(cell)
    cell_image = image[cell['top']: cell['bottom']+1, cell['left']: cell['right']+1]
    text, confidence = model.apply(cell_image)
    cell_output['text'] = text
    cell_output['confidence'] = confidence
    return cell_output



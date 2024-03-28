# library file
import json
import os.path

import cv2
import numpy as np

import geometry
from application import TrainedModel


def process_muster_sheet(input_jpg_file, image_metadata, output_dir, corrections_dir,
                         config, workflows, templates, exception_if_missing):
    """
    Process sheet...
    :param image_metadata:
    :param output_dir: directory for output files
    :param corrections_dir: directory for corrections for Zooniverse
    :param templates: template specifications
    :param input_jpg_file: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :param exception_if_missing: throw exception for missing file
    :return:
    """
    print(f'Output to {output_dir}')
    print(f'Reading image from {input_jpg_file}')
    format_type = image_metadata.get('format_type', '2')
    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]

    image_raw = safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                            exception_if_missing=exception_if_missing)
    template = templates[format_type]
    template_image = template['image']

    registered_image = geometry.adjust_muster_to_template(image_raw, template_image, config)
    previous_workflow_id = None
    model = None
    for sequence, cell in enumerate(template['cells']):
        workflow_id = cell['workflow']
        if workflow_id != previous_workflow_id:
            model_subdirectory = workflows[workflow_id]['model_location']
            print(f'Loading model from {model_subdirectory} for {workflow_id}')
            model = TrainedModel(model_subdirectory)

        threshold = workflows[workflow_id]['confidence']
        cell_result, cell_image = process_cell(registered_image, cell, model, threshold)
        cell_result.update(image_metadata)
        output_filename = os.path.join(output_dir,
                                       f'{output_prefix}_{sequence:04d}.json')
        print(f'Writing {output_filename}')
        with open(output_filename, 'w') as f:
            json.dump(cell_result, f)
        if config.debug:
            cell_image_filename = os.path.join(output_dir,
                                          f'{output_prefix}_{sequence:04d}.png')
            print(f'Writing {cell_image_filename}')
            cv2.imwrite(cell_image_filename, cell_image)

        # TODO generate image for checking if cell['confidence'] < threshold
        # TODO also generate metadata for Pantoptes and for reattaching
        # https://panoptes.docs.apiary.io/#
    return


def process_cell(image: np.ndarray, cell, model, threshold: float):
    cell_output = {'threshold': threshold}
    cell_output.update(cell)
    cell_image = image[cell['top']: cell['bottom'] + 1, cell['left']: cell['right'] + 1]
    print(f'Cell shape {cell_image.shape}')
    text, confidence = model.apply(cell_image)
    cell_output['text'] = text
    cell_output['confidence'] = confidence
    return cell_output, cell_image


def write_control_file(input_path: str, output_directory: str, data: dict):
    # not used yet
    input_name = os.path.basename(input_path)
    stem = os.path.splitext(input_name)[0]
    output_path = os.path.join(output_directory, stem + '.json')
    data['source'] = {'path': input_path,
                      'filename': input_name}
    print(f'Writing {output_path}')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return


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

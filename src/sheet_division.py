# library file
import json
import os.path

import cv2
import numpy as np

import geometry
from application import TrainedModel


def process_muster_sheet(input_jpg_file, image_metadata: dict, output_dir, corrections_dir,
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
    print(f'Reading image from {input_jpg_file}')
    format_type = image_metadata.get('format_type', '2')
    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]

    image_raw = safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                            exception_if_missing=exception_if_missing)
    template = templates[format_type]
    template_image = template['image']
    print(f'Input size {image_raw.shape}; template size {template_image.shape}')
    image_resized = geometry.resize_image(image_raw, template_image)
    print(f'Resized input size {image_resized.shape}')
    registered_image = geometry.adjust_muster_to_template(image_resized, template_image, config)

    if config['debug']:
        # superimpose the template in blue and save it for inspection
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_registration.png')
        registration_debug = geometry.superimpose(registered_image, template_image)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug)

    previous_workflow_id = None
    model = None
    for sequence, cell_spec in enumerate(template['cells']):
        workflow_id = cell_spec['workflow']
        print(f'Cell info: {cell_spec}')
        if workflow_id != previous_workflow_id:
            print()
            model_subdirectory = workflows[workflow_id]['model_location']
            print(f'Loading model from {model_subdirectory} for workflow {workflow_id}')
            model = TrainedModel(model_subdirectory)
            previous_workflow_id = workflow_id

        threshold = workflows[workflow_id]['confidence']
        cell_result, cell_image = process_cell(registered_image, cell_spec, model, threshold)
        cell_result['input'] = image_metadata.copy()
        save_processed_cell(config, output_prefix, sequence, cell_result, output_dir, cell_image)

        if cell_result['confidence'] < cell_result['threshold']:
            save_cell_for_correction(corrections_dir, output_prefix, sequence, cell_spec, cell_result, registered_image)
            # TODO also generate metadata for Pantoptes and for reattaching
            # https://panoptes.docs.apiary.io/#
    return


def save_cell_for_correction(correction_dir: str, output_prefix: str, sequence: int,
                             cell_spec: dict, cell_result: dict, big_image: np.ndarray):
    output_path = os.path.join(correction_dir, f'{output_prefix}_{sequence:04d}.png')
    output_image = geometry.generate_image_for_correction(big_image, cell_spec, cell_result['text'])
    print(f'Writing for correction {output_path}')
    cv2.imwrite(output_path, output_image)
    return


def save_processed_cell(config: dict, output_prefix: str, sequence:int, cell_result: dict,
                        output_dir: str, cell_image: np.ndarray):
    output_filename = os.path.join(output_dir,
                                   f'{output_prefix}_{sequence:04d}.json')
    print(f'Writing {output_filename}')
    # some of the stuff in cell_result is not json serializable e.g., np.float32
    with open(output_filename, 'w') as f:
        json.dump(cell_result, f, cls=NumpyJsonEncoder, indent=1, sort_keys=True)
    if config['debug']:
        cell_image_filename = os.path.join(output_dir,
                                           f'{output_prefix}_{sequence:04d}.png')
        cell_result['output_image'] = output_filename
        print(f'Writing {cell_image_filename}')
        cv2.imwrite(cell_image_filename, cell_image)
    return


def process_cell(image: np.ndarray, cell, model, threshold: float):
    """
    Extract the cell from the image and send the small image through the model.
    :param image: registered image of the whole sheet
    :param cell: specification for one cell
    :param model: trained model
    :param threshold: confidence score required for the workflow
    """
    cell_output = {'threshold': threshold}
    cell_output.update(cell)
    top, bottom = cell['top'], cell['bottom'] + 1
    left, right = cell['left'], cell['right'] + 1
    cell_image = image[top: bottom, left: right]
    text, confidence = model.apply(cell_image)
    cell_output['text'] = text
    cell_output['confidence'] = confidence
    print(f'Cell shape {cell_image.shape} confidence {confidence:.3f} / {threshold:.3f} text "{text}"')
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


class NumpyJsonEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    # custom encoder to get numpy types through
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJsonEncoder, self).default(obj)

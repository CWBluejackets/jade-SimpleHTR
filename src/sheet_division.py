# library file
import csv
import json
import os.path
from collections import defaultdict

import cv2
import numpy as np

import geometry
from application import TrainedModel


def process_muster_sheet(input_jpg_file, image_metadata: dict, output_dir, parts_dir, corrections_dir,
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
    try:
        form_type = image_metadata['type']
        print(f'SUMMARY: Format type is {form_type}')
    except KeyError:
        print(f'SUMMARY: Missing format_type!')
        return

    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]
    image_raw = safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                            exception_if_missing=exception_if_missing)
    template = templates[form_type]
    template_image = template['image']
    print(f'Input size {image_raw.shape}; template size {template_image.shape}')
    image_resized = geometry.resize_image(image_raw, template_image)

    if config['debug']:
        debug_image_path = os.path.join(output_dir, f'{output_prefix}_cutoffs.png')
    else:
        debug_image_path = None

    registered_image = geometry.adjust_muster_to_template(image_resized, template_image,
                                                          config, debug_image_path)

    csv_header = ['row']
    for column in template['output_columns']:
        csv_header.extend([str(column['workflow']), 'conf'])
    csv_table = defaultdict(lambda: [str()] * len(template['output_columns'] * 2))

    if config['debug']:
        # superimpose the template in blue and save it for inspection
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_registration.png')
        registration_debug = geometry.superimpose(registered_image, template_image)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug)

    previous_workflow_id = None
    model = None
    for cell_spec in template['cells']:
        workflow_id = cell_spec['workflow']
        print(f'Cell info: {cell_spec}')
        row = cell_spec['row']
        column = cell_spec['column']
        filename_base = f'{output_prefix}_{row:02d}_{column:02d}'

        if workflow_id != previous_workflow_id:
            print()
            model_subdirectory = workflows[workflow_id]['model_location']
            print(f'Loading model from {model_subdirectory} for workflow {workflow_id}')
            model = TrainedModel(model_subdirectory)
            previous_workflow_id = workflow_id

        cell_result, cell_image = process_cell(registered_image, cell_spec, model)
        cell_result['input'] = image_metadata.copy()
        save_processed_cell(config, filename_base, cell_result, parts_dir, cell_image)
        csv_table[cell_result['row']][cell_result['column'] * 2] = cell_result['text']
        csv_table[cell_result['row']][cell_result['column'] * 2 + 1] = cell_result['confidence']

        if cell_result['confidence'] < cell_result['confidence_threshold']:
            save_cell_for_correction(corrections_dir, filename_base, cell_spec,
                                     cell_result, registered_image)
            # TODO also generate metadata for Pantoptes and for reattaching
            # https://panoptes.docs.apiary.io/#

    csv_file = os.path.join(output_dir, f'{output_prefix}.csv')
    print(f'Writing {csv_file}')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for row_no in range(max(csv_table.keys())):
            writer.writerow([row_no] + csv_table[row_no])
    return


def save_cell_for_correction(correction_dir: str, filename_base: str,
                             cell_spec: dict, cell_result: dict, big_image: np.ndarray):
    output_image_path = os.path.join(correction_dir, f'{filename_base}.png')
    output_image = geometry.generate_image_for_correction(big_image, cell_spec, cell_result['text'])
    print(f'Writing for correction {output_image_path}')
    cv2.imwrite(output_image_path, output_image)
    output_json_path = os.path.join(correction_dir, f'{filename_base}.json')
    print(f'Writing for correction {output_json_path}')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cell_result, f, indent=1, sort_keys=True)
    return


def save_processed_cell(config: dict, filename_base: str, cell_result: dict,
                        output_dir: str, cell_image: np.ndarray):
    output_filename = os.path.join(output_dir,
                                   f'{filename_base}.json')
    print(f'Writing {output_filename}')
    # some of the stuff in cell_result is not json serializable e.g., np.float32
    with open(output_filename, 'w') as f:
        json.dump(cell_result, f, cls=NumpyJsonEncoder, indent=1, sort_keys=True)
    if config['debug']:
        cell_image_filename = os.path.join(output_dir,
                                           f'{filename_base}.png')
        cell_result['output_image'] = output_filename
        print(f'Writing {cell_image_filename}')
        cv2.imwrite(cell_image_filename, cell_image)
    return


def process_cell(image: np.ndarray, cell, model):
    """
    Extract the cell from the image and send the small image through the model.
    :param image: registered image of the whole sheet
    :param cell: specification for one cell
    :param model: trained model
    """
    top, bottom = cell['top'], cell['bottom'] + 1
    left, right = cell['left'], cell['right'] + 1
    cell_image = image[top: bottom, left: right]
    text, confidence = model.apply(cell_image)
    cell_output = {'text': text, 'confidence': confidence}
    cell_output.update(cell)
    threshold = cell['confidence_threshold']
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

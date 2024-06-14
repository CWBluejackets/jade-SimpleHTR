# library file
import csv
import json
import os.path
from collections import defaultdict

import cv2
import numpy as np

import geometry
from application import TrainedModel
from utils import NumpyJsonEncoder, EIGHTHS, RED
from cellulation import generate_cells


def process_muster_sheet(input_jpg_file, image_metadata: dict, output_dir, parts_dir, corrections_dir,
                         config, workflows, templates):
    """
    Process sheet...
    :param image_metadata:
    :param output_dir: directory for output files
    :param parts_dir: directory for parts files
    :param corrections_dir: directory for corrections for Zooniverse
    :param templates: template specifications
    :param input_jpg_file: JPG filename
    :param config: general configuration
    :param workflows: workflow specifications
    :return:
    """
    print(f'Reading image from {input_jpg_file}')
    try:
        form_type = image_metadata['type']
        print(f'SUMMARY: Form type is {form_type}')
    except KeyError:
        print(f'SUMMARY: Missing form type!')
        return
    if form_type not in templates:
        print(f'SUMMARY: Form type {form_type} is not supported')
        return

    try:
        image_raw = safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                exception_if_missing=True)
    except FileNotFoundError:
        print(f'SUMMARY: File {input_jpg_file} not found')
        return

    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]
    template = templates[form_type]
    # now trying the top of the form
    template_image = template['image_top']
    template_form_width = template['form_width']
    print(f'Input size {image_raw.shape}; template size {template_image.shape}')
    sheet_info = {'input_file': input_jpg_file,
                  'form_type': form_type,
                  'template_form_width': template_form_width,
                  'shape': {'raw_input': image_raw.shape}}
    image_resized = geometry.resize_image(image_raw, template_image)
    registered_image = geometry.adjust_muster_to_template(image_resized, template_image,
                                                          template_form_width,
                                                          config, output_dir, output_prefix,
                                                          sheet_info)
    csv_header = ['row']
    for column in template['output_columns']:
        csv_header.extend([str(column['workflow']), 'conf'])
    csv_table = defaultdict(lambda: [str()] * len(template['output_columns'] * 2))

    hline_y_min = template['hline_detection']['y_min']
    hline_y_max = template['hline_detection']['y_max']
    y_ref_line = template['y_ref_line']
    row_y_values = geometry.find_hlines(registered_image, config, hline_y_min, hline_y_max, y_ref_line)
    sheet_info['row_detection'] = {'y_values': row_y_values,
                                   'diffs': row_y_values[1:] - row_y_values[:-1]}
    if config['debug']:
        # superimpose the template in blue and save it for inspection
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_2_registration.png')
        registration_debug_2 = geometry.superimpose(registered_image, template_image)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug_2)
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_3_registration.png')
        full_template_image = template['image']
        registration_debug_3 = geometry.superimpose(registered_image, full_template_image)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug_3)
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_4_registration.png')
        x_left = min(template['output_column_x_values'])
        x_right = max(template['output_column_x_values'])
        y_top = min(row_y_values)
        y_bottom = max(row_y_values)
        for y_value in row_y_values:
            cv2.line(registration_debug_2, (x_left, y_value), (x_right, y_value), RED, 2)
        for x_value in template['output_column_x_values']:
            cv2.line(registration_debug_2, (x_value, y_top), (x_value, y_bottom), RED, 2)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug_2)

    # produce median data for rows
    row_data = dict()
    for row_number, row_spec in template['rows'].items():
        row_info = row_spec.copy()
        row_key = f'{row_number:02d}'
        row_info['median'] = np.median(registered_image[row_spec['top']: row_spec['bottom'],
                                       row_spec['left']: row_spec['right']])
        row_data[row_key] = row_info
    sheet_info['rows'] = row_data

    previous_workflow_id = None
    model = None
    confidence_by_workflow = defaultdict(list)

    cell_list = generate_cells(template, workflows, row_y_values)
    for cell_spec in cell_list:
        workflow_id = cell_spec['workflow']
        row = cell_spec['row']
        column = cell_spec['column']
        filename_base = f'{output_prefix}_{row:02d}_{column:02d}'

        if config['htr'] and (workflow_id != previous_workflow_id):
            print()
            model_subdirectory = workflows[workflow_id]['model_location']
            print(f'Loading model from {model_subdirectory} for workflow {workflow_id}')
            model = TrainedModel(model_subdirectory)
            previous_workflow_id = workflow_id

        cell_result, cell_image = process_cell(registered_image, cell_spec, model, config)
        cell_result['input'] = image_metadata.copy()
        cell_result['pcc'] = sheet_info['pcc']
        save_processed_cell(config, filename_base, cell_result, parts_dir, cell_image)
        if config['htr']:
            csv_table[cell_result['row']][cell_result['column'] * 2] = cell_result['text']
            csv_table[cell_result['row']][cell_result['column'] * 2 + 1] = cell_result['confidence']
            confidence_by_workflow[workflow_id].append(cell_result['confidence'])

        if config['htr'] and (cell_result['confidence'] < cell_result['confidence_threshold']):
            add_training_boundaries(cell_result)
            workflow_name = workflows[cell_spec['workflow']]['description']
            save_cell_for_correction(corrections_dir, filename_base, cell_spec,
                                     cell_result, registered_image, workflow_name)
            # TODO also generate metadata for Pantoptes and for reattaching
            # https://panoptes.docs.apiary.io/#

    if config['htr']:
        csv_file = os.path.join(output_dir, f'{output_prefix}.csv')
        print(f'SUMMARY: Writing {csv_file}')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for row_no in range(max(csv_table.keys())):
                writer.writerow([row_no] + csv_table[row_no])
        confidence_data = dict()
        for workflow_id, confidences in confidence_by_workflow.items():
            confidence_data[str(workflow_id)] = {'quantiles': np.quantile(confidences, EIGHTHS),
                                                 'count': len(confidences)}
        all_confidences = sum(confidence_by_workflow.values(), start=[])
        confidence_data['all'] = {'quantiles': np.quantile(all_confidences, EIGHTHS),
                                  'count': len(all_confidences)}
        sheet_info['confidences'] = confidence_data

    sheet_info_file = os.path.join(output_dir, f'{output_prefix}_info.json')
    print(f'SUMMARY: Writing {sheet_info_file}')
    with open(sheet_info_file, 'w') as f:
        json.dump(sheet_info, f, indent=1, sort_keys=True, cls=NumpyJsonEncoder)
    return


def save_cell_for_correction(correction_dir: str, filename_base: str,
                             cell_spec: dict, cell_result: dict,
                             big_image: np.ndarray, workflow_name: str):
    output_image_path = os.path.join(correction_dir, f'{filename_base}.png')
    output_image = geometry.generate_image_for_correction(big_image, cell_spec,
                                                          cell_result['text'], workflow_name)
    cell_spec['output_for_correction'] = output_image_path
    print(f'Writing for correction {output_image_path}')
    cv2.imwrite(output_image_path, output_image)
    output_json_path = os.path.join(correction_dir, f'{filename_base}.json')
    print(f'Writing for correction {output_json_path}')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cell_result, f, cls=NumpyJsonEncoder, indent=1, sort_keys=True)
    return


def add_training_boundaries(cell_result: dict) -> None:
    """
    Add the coordinates for extracting the image for HTR for reuse as
    training data after correction.
    :param cell_result: accumulated stuff about the cell
    """
    top = cell_result['htr']['top'] - cell_result['expanded']['top']
    bottom = cell_result['htr']['bottom'] + cell_result['expanded']['top']
    left = cell_result['htr']['left'] - cell_result['expanded']['left']
    right = cell_result['htr']['right'] + cell_result['expanded']['left']
    cell_result['training_boundaries'] = {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }
    return


def save_processed_cell(config: dict, filename_base: str, cell_result: dict,
                        output_dir: str, cell_image: np.ndarray):
    output_filename = os.path.join(output_dir,
                                   f'{filename_base}.json')
    with open(output_filename, 'w') as f:
        json.dump(cell_result, f, cls=NumpyJsonEncoder, indent=1, sort_keys=True)
    if config['debug']:
        cell_image_filename = os.path.join(output_dir,
                                           f'{filename_base}.png')
        cell_result['output_image'] = output_filename
        cv2.imwrite(cell_image_filename, cell_image)
    return


def process_cell(image: np.ndarray, cell, model, config: dict):
    """
    Extract the cell from the image and send the small image through the model.
    :param image: registered image of the whole sheet
    :param cell: specification for one cell
    :param model: trained model
    :param config: configuration dictionary
    """
    inner_top, inner_bottom = cell['original']['top'], cell['original']['bottom']
    inner_left, inner_right = cell['original']['left'], cell['original']['right']
    top, bottom = cell['htr']['top'], cell['htr']['bottom']
    left, right = cell['htr']['left'], cell['htr']['right']
    inner_cell_image = image[inner_top:inner_bottom, inner_left:inner_right]
    cell_image = image[top: bottom, left: right]
    cell_output = dict()
    if config['htr']:
        text, confidence = model.apply(cell_image)
        cell_output.update({'text': text, 'confidence': confidence})
        threshold = cell['confidence_threshold']
        print(f'Cell shape {cell_image.shape} confidence {confidence:.3f} / {threshold:.3f} text "{text}"')
    cell_output.update(cell)
    cell_output['median_full'] = np.median(cell_image)
    cell_output['median_inner'] = np.median(inner_cell_image)
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
        json.dump(data, f, cls=NumpyJsonEncoder, indent=2, sort_keys=True)
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

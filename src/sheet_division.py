# library file
import csv
import json
import os.path
import random
from collections import defaultdict

import cv2
import numpy as np

import geometry
from application import TrainedModel
from utils import NumpyJsonEncoder, EIGHTHS, RED, BLUE, safe_imread, write_sheet_info
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
    sheet_info = {'input_file': input_jpg_file}
    output_prefix = os.path.splitext(os.path.basename(input_jpg_file))[0]

    try:
        form_type = image_metadata['type']
        sheet_info['form_type'] = form_type
        print(f'SUMMARY: Form type is {form_type}')
    except KeyError:
        print(f'SUMMARY: Missing form type!')
        sheet_info['form_type'] = '_MISSING_'
        write_sheet_info(output_dir, output_prefix, sheet_info)
        return
    if form_type not in templates:
        print(f'SUMMARY: Form type {form_type} is not supported')
        sheet_info['error'] = 'Form type is not supported'
        write_sheet_info(output_dir, output_prefix, sheet_info)
        return

    try:
        image_raw = safe_imread(input_jpg_file, cv2.IMREAD_GRAYSCALE,
                                exception_if_missing=True)
    except FileNotFoundError:
        print(f'SUMMARY: File {input_jpg_file} not found')
        sheet_info['error'] = 'File not found'
        write_sheet_info(output_dir, output_prefix, sheet_info)
        return

    template = templates[form_type]
    # now trying the top of the form
    template_image = template['image_top']
    template_form_width = template['form_width']
    print(f'Input size {image_raw.shape}; template size {template_image.shape}')
    sheet_info.update({'template_form_width': template_form_width,
                       'shape': {'raw_input': image_raw.shape}})
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
    row_y_values, horizontal_means = geometry.find_row_boundaries(registered_image, config, hline_y_min,
                                                                  hline_y_max, y_ref_line)
    sheet_info['row_detection'] = {'y_values': row_y_values,
                                   'diffs': row_y_values[1:] - row_y_values[:-1]}
    x_left = min(template['output_column_x_values'])
    x_right = max(template['output_column_x_values'])
    if config['debug']:
        registration_path = os.path.join(output_dir,
                                         f'{output_prefix}_2_registration.png')
        # superimpose the header template in blue and save it for inspection
        registration_debug_2 = geometry.superimpose(registered_image, template_image)
        # superimpose the lines detected and the whiteness plot
        y_top = min(row_y_values)
        y_bottom = max(row_y_values)
        # mark the horizontal lines in red
        for y_value in row_y_values:
            cv2.line(registration_debug_2, (x_left, y_value), (x_right, y_value), RED, 2)
        # mark the extended columns in red
        for x_value in template['output_column_x_values']:
            cv2.line(registration_debug_2, (x_value, y_top), (x_value, y_bottom), RED, 2)
        # add the horizontal means for inspection on the left
        for y, h_mean in enumerate(horizontal_means):
            cv2.line(registration_debug_2, (0, y), (round(h_mean * 3), y), RED, 1)
        # mark the bottom limit from the template in blue
        cv2.line(registration_debug_2, (255 * 3, hline_y_max), (x_right, hline_y_max), BLUE, 1)
        print(f'Writing {registration_path}')
        cv2.imwrite(registration_path, registration_debug_2)

    previous_workflow_id = None
    model = None
    confidence_by_workflow = defaultdict(list)

    cell_list = generate_cells(template, workflows, row_y_values)

    # produce median data for rows
    row_data = dict()
    for row_number, (top, bottom) in enumerate(zip(row_y_values[:-1], row_y_values[1:])):
        row_key = f'{row_number:02d}'
        row_image = registered_image[top: bottom, x_left: x_right]
        row_info = {'top': top,
                    'bottom': bottom,
                    'octiles': np.quantile(row_image, EIGHTHS),
                    'mean': row_image.mean()}
        row_data[row_key] = row_info
    sheet_info['rows'] = row_data

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

        if config['debug']:
            save_processed_cell(config, filename_base, cell_result, parts_dir, cell_image)

        if config['htr']:
            csv_table[cell_result['row']][cell_result['column'] * 2] = cell_result['text']
            csv_table[cell_result['row']][cell_result['column'] * 2 + 1] = cell_result['confidence']
            confidence_by_workflow[workflow_id].append(cell_result['confidence'])

        elif random.random() < 0.02:
            # if not running HTR, generate a random 2% of correction images to check the geometry
            add_training_boundaries(cell_result)
            workflow_name = workflows[cell_spec['workflow']]['description']
            cell_result['text'] = ' '.join(['Testing'] * random.randint(1, 4))
            cell_result['confidence'] = random.uniform(0.0, cell_result['confidence_threshold'])
            save_cell_for_correction(corrections_dir, filename_base, cell_spec,
                                     cell_result, registered_image, workflow_name)

        if config['htr'] and (cell_result['confidence'] < cell_result['confidence_threshold']):
            add_training_boundaries(cell_result)
            workflow_name = workflows[cell_spec['workflow']]['description']
            save_cell_for_correction(corrections_dir, filename_base, cell_spec,
                                     cell_result, registered_image, workflow_name)

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
            confidence_data[str(workflow_id)] = {'octiles': np.quantile(confidences, EIGHTHS),
                                                 'count': len(confidences)}
        all_confidences = sum(confidence_by_workflow.values(), start=[])
        confidence_data['all'] = {'octiles': np.quantile(all_confidences, EIGHTHS),
                                  'count': len(all_confidences)}
        sheet_info['confidences'] = confidence_data

    write_sheet_info(output_dir, output_prefix, sheet_info)
    return


def save_cell_for_correction(correction_dir: str, filename_base: str,
                             cell_spec: dict, cell_result: dict,
                             big_image: np.ndarray, workflow_name: str):
    output_image_path = os.path.join(correction_dir, 'image', f'{filename_base}.jpg')
    output_image = geometry.generate_image_for_correction(big_image, cell_spec,
                                                          cell_result['text'], workflow_name)
    cell_result['correction_image'] = output_image_path
    cell_result['workflow_name'] = workflow_name
    print(f'Writing for correction {output_image_path}')
    cv2.imwrite(output_image_path, output_image)
    output_json_path = os.path.join(correction_dir, 'json', f'{filename_base}.json')
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
    bottom = cell_result['htr']['bottom'] - cell_result['expanded']['top']
    left = cell_result['htr']['left'] - cell_result['expanded']['left']
    right = cell_result['htr']['right'] - cell_result['expanded']['left']
    cell_result['training_boundaries'] = {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }
    return


def save_processed_cell(config: dict, filename_base: str, cell_result: dict,
                        parts_dir: str, cell_image: np.ndarray):
    output_filename = os.path.join(parts_dir,
                                   f'{filename_base}.json')
    with open(output_filename, 'w') as f:
        json.dump(cell_result, f, cls=NumpyJsonEncoder, indent=1, sort_keys=True)
    if config['debug']:
        cell_image_filename = os.path.join(parts_dir,
                                           f'{filename_base}.jpg')
        cell_result['debug_image'] = cell_image_filename
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
    threshold = cell['confidence_threshold']
    if config['htr']:
        text, confidence = model.apply(cell_image)
        cell_output.update({'text': text, 'confidence': confidence})
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

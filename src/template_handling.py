import json
import os
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
from statistics import median

import sheet_division


def find_lines(scanline: np.ndarray) -> Tuple[List[int], int]:
    """
    Scan a row/column for contiguous black (0) bits and return the middle position of each
    group.
    :param scanline: array of rank 1
    :return: list of middle positions of black lines AND overall width for rescaling (ignore
    this for row)
    """
    black = np.where(scanline == 0)[0]
    groups = []
    previous = black[0]
    group = [previous]
    for pixel in black[1:]:
        if pixel == previous + 1:
            group.append(pixel)
        else:
            groups.append(group)
            group = [pixel]
        previous = pixel
    groups.append(group)
    middles = [round(median(group)) for group in groups]
    form_width = max(black) - min(black)
    return middles, form_width


def get_column_separators(form: np.ndarray, y_ref: int):
    """
    Find the black column lines in the first writing row of the form
    :param form: form image
    :param y_ref: y-value of the line at the top of the first writing row
    :return: list of x-values of black column separator lines
    AND x-value of the middle of the first column
    AND overall width of the form
    """
    scanline = form[y_ref + 10, :]
    middles, form_width = find_lines(scanline)
    x_ref = round((middles[0] + middles[1]) / 2)
    return middles, x_ref, form_width


def get_row_separators(form: np.ndarray, y_ref: int, x_ref: int):
    """
    Find the black row lines after the top of the first writing row of the form
    :param form: form image
    :param y_ref: y-value of the line at the top of the first writing row
    :param x_ref: x-value in the middle of the first column
    :return:
    """
    y_offset = y_ref + 10
    scanline = form[y_offset:, x_ref]
    shifted_rows, _ = find_lines(scanline)
    return [row + y_offset for row in shifted_rows]


def add_template_info(specification: dict, template_directory, workflows, debug_template_path):
    """
    Load the template image; derive row and column information; derive cell boundaries
    :param specification: a value from pipeline-templates.json
    :return: enhanced version of the same
    :param template_directory: location of the templates
    :param workflows: workflow specifications
    """
    template_path = os.path.join(template_directory, specification['template'])
    template_top_path = os.path.join(template_directory, specification['template_top'])
    specification['template_path'] = template_path
    specification['template_top_path'] = template_top_path
    y_ref_line = specification['y_ref_line']
    print(f'Loading template image from {template_path}')
    image = sheet_division.safe_imread(template_path, cv2.IMREAD_GRAYSCALE)
    print(f'Loading template top image from {template_top_path}')
    image_top = sheet_division.safe_imread(template_top_path, cv2.IMREAD_GRAYSCALE)
    # find the black column boundaries in the template
    column_lines, x_ref_line, form_width = get_column_separators(image, y_ref_line)
    specification['form_width'] = form_width
    columns_needed = len(specification['columns'])
    if columns_needed != len(column_lines) - 1:
        raise ValueError(f'Mismatch! specified columns = {columns_needed}; boundaries found = {len(column_lines)}')
    # find the black row lines (starting with the bottom of the first writing row
    # because y_ref_line, the top of the first writing row, is given)
    row_lines = [y_ref_line] + get_row_separators(image, y_ref_line, x_ref_line)
    specification['row_lines'] = row_lines
    specification['max_y'] = image.shape[0] - 1
    specification['max_x'] = image.shape[1] - 1
    # These pre-specified columns are more numerous than our workflows
    for index, column in enumerate(specification['columns']):
        column['start'] = column_lines[index]
        column['end'] = column_lines[index + 1]
    # Merge adjacent columns of the same workflow (e.g., height feet + height inches
    # and determine the start and end x-values (this code relies on the adjacency of parts
    # of the same workflow)
    workflow_starts = defaultdict(set)
    workflow_ends = defaultdict(set)
    for column in specification['columns']:
        if column['process']:
            workflow = column['workflow']
            workflow_starts[workflow].add(column['start'])
            workflow_ends[workflow].add(column['end'])
    output_columns = []
    for workflow in workflow_starts.keys():
        output_column = {'workflow': workflow,
                         'start': min(workflow_starts[workflow]),
                         'end': max(workflow_ends[workflow])}
        output_columns.append(output_column)
    specification['output_columns'] = output_columns
    generate_cells(specification, workflows)
    generate_rows(specification)
    # save the debug info WITHOUT the big image arrays
    if debug_template_path:
        print(f'Writing {debug_template_path}')
        with open(debug_template_path, 'w') as f:
            json.dump(specification, f, indent=1, sort_keys=True, cls=sheet_division.NumpyJsonEncoder)
    # now add the arrays
    specification['image'] = image
    specification['image_top'] = image_top
    specification['hline_detection'] = {'y_min': y_ref_line + 5,
                                        'y_max': row_lines[-1] + 5}
    return


def generate_rows(specification: dict):
    row_boundaries = specification['row_lines']
    left = specification['output_columns'][0]['start'] + 1
    right = specification['output_columns'][-1]['end'] - 1
    row_data = {}
    for row_number, (top, bottom) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
        row_data[row_number] = {"top": top,
                                "bottom": bottom,
                                "left": left,
                                "right": right}
    specification['rows'] = row_data
    pass


def generate_cells(specification: dict, workflows: dict):
    """
    Add all the individual cell details to the template specification.
    :param specification: the template to be updated
    :param workflows: workflow specifications
    """
    cells = []
    for column_number, column_details in enumerate(specification['output_columns']):
        workflow = column_details['workflow']
        threshold = workflows[workflow]['confidence']
        left_adj, right_adj, top_adj, bottom_adj = get_cell_adjustments(workflow, workflows)
        original_left = column_details['start']
        htr_left = column_details['start']
        original_right = column_details['end']
        htr_right = column_details['end']
        expanded_left, expanded_right = get_expanded_x_values(column_number, specification['output_columns'])
        width = htr_right - htr_left
        # 0 do nothing
        # 0.0 < ADJ < 1.0 is a proportion of height or width
        # 1 up is number of pixels
        if 0 < left_adj < 1:
            htr_left = htr_left - round(left_adj * width)
        elif 1 <= left_adj:
            htr_left = htr_left - left_adj
        if 0 < right_adj < 1:
            htr_right = htr_right + round(right_adj * width)
        elif 1 <= right_adj:
            htr_right = htr_right + right_adj
        row_boundaries = specification['row_lines']
        for row_number, (top, bottom) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
            original_top = top
            htr_top = top
            original_bottom = bottom
            htr_bottom = bottom
            expanded_top, expanded_bottom = get_expanded_y_values(row_number, row_boundaries)
            height = htr_bottom - htr_top
            if 0 < top_adj < 1:
                htr_top = htr_top - round(top_adj * height)
            elif 1 <= top_adj:
                htr_top = htr_top - top_adj
            if 0 < bottom_adj < 1:
                htr_bottom = htr_bottom + round(bottom_adj * height)
            elif 1 <= bottom_adj:
                htr_bottom = htr_bottom + bottom_adj
            cell = {"workflow": workflow,
                    "confidence_threshold": threshold,
                    "column": column_number,
                    "row": row_number,
                    "htr": {
                        "left": htr_left,
                        "right": htr_right,
                        "top": htr_top,
                        "bottom": htr_bottom,
                    },
                    "original": {"left": original_left,
                                 "right": original_right,
                                 "top": original_top,
                                 "bottom": original_bottom
                                 },
                    "expanded": {"top": max(expanded_top, 0),
                                 "bottom": min(expanded_bottom, specification['max_y']),
                                 "left": max(expanded_left, 0),
                                 "right": min(expanded_right, specification['max_x'])
                                 }
                    }
            cells.append(cell)
    specification['cells'] = cells
    return


def get_cell_adjustments(workflow, workflows: dict):
    """
    Get the cell expansions from the workflow file. Use 0 wherever
    nothing is specified.
    """
    adjustments = workflows[workflow].get('extras', dict())
    left = adjustments.get('left', 0)
    right = adjustments.get('right', 0)
    top = adjustments.get('top', 0)
    bottom = adjustments.get('bottom', 0)
    return left, right, top, bottom


def get_expanded_x_values(column_index: int, columns_info) -> Tuple[int, int]:
    """
    Get x-values to cover one column left and right of the specified one, or
    just some pixels at the ends.
    """
    if column_index == 0:
        left = columns_info[column_index]['start'] - 100
    else:
        left = columns_info[column_index - 1]['start']
    if column_index == len(columns_info) - 1:
        right = columns_info[column_index]['end'] + 100
    else:
        right = columns_info[column_index + 1]['end']
    return left, right + 1


def get_expanded_y_values(row_index: int, row_boundaries: List[int]) -> Tuple[int, int]:
    """
    Get y-values to cover two rows above and below the specified row, or just some pixels
    when there are not two rows to get.
    """
    if row_index == 0:
        top = row_boundaries[0] - 120
    elif row_index == 1:
        top = row_boundaries[0] - 100
    else:
        top = row_boundaries[row_index - 2]
    if row_index == len(row_boundaries) - 1:
        bottom = row_boundaries[-1] + 120
    elif row_index == len(row_boundaries) - 2:
        bottom = row_boundaries[-1] + 100
    else:
        bottom = row_boundaries[row_index + 2]
    return top, bottom + 1

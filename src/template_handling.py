import json
import os
from collections import defaultdict
from statistics import median
from typing import List, Tuple

import cv2
import numpy as np

from utils import safe_imread, NumpyJsonEncoder


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
    image = safe_imread(template_path, cv2.IMREAD_GRAYSCALE)
    print(f'Loading template top image from {template_top_path}')
    image_top = safe_imread(template_top_path, cv2.IMREAD_GRAYSCALE)
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
    # following to be used for automatic row breaking
    specification['hline_detection'] = {'y_min': y_ref_line + 5,
                                        'y_max': row_lines[-1] + 5}
    # These pre-specified columns are more numerous than our workflows
    for index, column in enumerate(specification['columns']):
        column['start'] = column_lines[index]
        column['end'] = column_lines[index + 1]
    # Merge adjacent columns of the same workflow (e.g., height feet + height inches
    # and determine the start and end x-values (this code relies on the adjacency of parts
    # of the same workflow)
    workflow_starts = defaultdict(set)
    workflow_ends = defaultdict(set)
    column_x_values = set()
    output_column_x_values = set()
    for column in specification['columns']:
        if column['process']:
            workflow = column['workflow']
            workflow_starts[workflow].add(column['start'])
            workflow_ends[workflow].add(column['end'])
            column_x_values.add(column['start'])
            column_x_values.add(column['end'])
    output_columns = []
    for workflow in workflow_starts.keys():
        start = min(workflow_starts[workflow])
        end = max(workflow_ends[workflow])
        output_column = {'workflow': workflow,
                         'start': start,
                         'end': end}
        output_columns.append(output_column)
        output_column_x_values |= {start, end}
    specification['output_columns'] = output_columns
    specification['column_x_values'] = column_x_values
    specification['output_column_x_values'] = output_column_x_values
    # save the debug info WITHOUT the big image arrays
    if debug_template_path:
        print(f'Writing {debug_template_path}')
        with open(debug_template_path, 'w') as f:
            json.dump(specification, f, indent=1, sort_keys=True, cls=NumpyJsonEncoder)
    # now add the arrays
    specification['image'] = image
    specification['image_top'] = image_top
    return



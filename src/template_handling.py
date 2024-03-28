import os
from collections import defaultdict
from typing import List

import cv2
import numpy as np
from statistics import median

import sheet_division


def find_lines(scanline: np.ndarray) -> List[int]:
    """
    Scan a row/column for contiguous black (0) bits and return the middle position of each
    group.
    :param scanline: array of rank 1
    :return: list of middle positions of black lines
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
    return middles


def get_columns(form: np.ndarray, y_ref: int):
    """
    Find the black column lines in the first writing row of the form
    :param form: form image
    :param y_ref: y-value of the line at the top of the first writing row
    :return: list of x-values of black lines, x-value of the middle of the first column
    """
    scanline = form[y_ref + 10, :]
    middles = find_lines(scanline)
    x_ref = round((middles[0] + middles[1]) / 2)
    return middles, x_ref


def get_rows(form: np.ndarray, y_ref: int, x_ref: int):
    """
    Find the black row lines after the top of the first writing row of the form
    :param form: form image
    :param y_ref: y-value of the line at the top of the first writing row
    :param x_ref: x-value in the middle of the first column
    :return:
    """
    y_offset = y_ref + 10
    scanline = form[y_offset:, x_ref]
    shifted_rows = find_lines(scanline)
    return [row + y_offset for row in shifted_rows]


def add_template_info(specification: dict, template_directory, workflows):
    """
    Load the template image; derive row and column information; derive cell boundaries
    :param specification: a value from pipeline-templates.json
    :return: enhanced version of the same
    :param template_directory: location of the templates
    :param workflows: workflow specifications
    """
    template_file = os.path.join(template_directory, specification['template'])
    y_ref_line = specification['y_ref_line']
    print(f'Loading template image from {template_file}')
    image = sheet_division.safe_imread(template_file, cv2.IMREAD_GRAYSCALE)
    # find the black column boundaries in the template
    column_lines, x_ref_line = get_columns(image, y_ref_line)
    columns_needed = len(specification['columns'])
    if columns_needed != len(column_lines) - 1:
        raise ValueError(f'Mismatch! specified columns = {columns_needed}; boundaries found = {len(column_lines)}')
    # find the black row lines (starting with the bottom of the first writing row)
    row_lines = [y_ref_line] + get_rows(image, y_ref_line, x_ref_line)
    specification['image'] = image
    specification['row_lines'] = row_lines
    for index, column in enumerate(specification['columns']):
        column['start'] = column_lines[index]
        column['end'] = column_lines[index + 1]
    # merge adjacent columns of the same workflow (e.g., height feet + height inches
    # and determine the start and end x-values
    output_columns = []
    workflow_starts = defaultdict(set)
    workflow_ends = defaultdict(set)
    for column in specification['columns']:
        if column['process']:
            workflow = column['workflow']
            workflow_starts[workflow].add(column['start'])
            workflow_ends[workflow].add(column['end'])
    for workflow in workflow_starts.keys():
        column = {'workflow': workflow,
                  'start': min(workflow_starts),
                  'end': max(workflow_ends)}
        output_columns.append(column)
    specification['output_columns'] = output_columns
    generate_cells(specification, workflows)
    return


def generate_cells(specification: dict, workflows: dict):
    cells = []
    for column_number, column in enumerate(specification['output_columns']):
        workflow = column['workflow']
        left_adj, right_adj, top_adj, bottom_adj = get_cell_adjustments(workflow, workflows)
        left = column['start']
        right = column['end']
        width = right - left
        # 0 do nothing
        # float is a proportion of height or width
        # 1 up is number of pixels
        if 0 < left_adj < 1:
            left = left - round(left_adj * width)
        elif 1 <= left_adj:
            left = left - left_adj
        if 0 < right_adj < 1:
            right = right + round(right_adj * width)
        elif 1 <= right_adj:
            right = right + right_adj
        row_boundaries = specification['row_lines']
        for row_number, (top, bottom) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
            height = bottom - top
            if 0 < top_adj < 1:
                top = top - round(top_adj * height)
            elif 1 <= top_adj:
                top = top - top_adj
            if 0 < bottom_adj < 1:
                bottom = bottom + round(bottom_adj * height)
            elif 1 <= bottom_adj:
                bottom = bottom + bottom_adj
            cell = {"workflow": workflow,
                    "column": column_number,
                    "row": row_number,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom}
            cells.append(cell)
    specification['cells'] = cells
    return


def get_cell_adjustments(workflow, workflows: dict):
    """
    Get the cell expansions from the workflow file
    """
    adjustments = workflows[workflow].get('extras', dict())
    left = adjustments.get('left', 0)
    right = adjustments.get('right', 0)
    top = adjustments.get('top', 0)
    bottom = adjustments.get('bottom', 0)
    return left, right, top, bottom


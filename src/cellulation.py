from typing import List, Tuple

import numpy as np


def generate_cells(template: dict, workflows: dict, row_boundaries: np.ndarray) -> list:
    columns = template['output_columns']
    # -> start, end, workflow
    cells = []
    for column_no, column_details in enumerate(columns):
        workflow = column_details['workflow']
        threshold = workflows[workflow]['confidence']
        left_adj, right_adj, top_adj, bottom_adj = get_cell_adjustments(workflow, workflows)
        original_left = column_details['start']
        htr_left = column_details['start']
        original_right = column_details['end']
        htr_right = column_details['end']
        expanded_left, expanded_right = get_expanded_x_values(column_no, columns)
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
        for row_number, (top, bottom) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
            original_top = top
            htr_top = top
            original_bottom = bottom
            htr_bottom = bottom
            expanded_top, expanded_bottom = get_expanded_y_values(row_number, row_boundaries.tolist())
            height = htr_bottom - htr_top
            # do not expand the top row upwards (into the headers)
            if row_number == 0:
                pass
            elif 0 < top_adj < 1:
                htr_top = htr_top - round(top_adj * height)
            elif 1 <= top_adj:
                htr_top = htr_top - top_adj
            if 0 < bottom_adj < 1:
                htr_bottom = htr_bottom + round(bottom_adj * height)
            elif 1 <= bottom_adj:
                htr_bottom = htr_bottom + bottom_adj
            cell = {"workflow": workflow,
                    "confidence_threshold": threshold,
                    "column": column_no,
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
                                 "bottom": min(expanded_bottom, template['max_y']),
                                 "left": max(expanded_left, 0),
                                 "right": min(expanded_right, template['max_x'])
                                 }
                    }
            cells.append(cell)
    return cells


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


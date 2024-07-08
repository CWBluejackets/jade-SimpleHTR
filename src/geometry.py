# library file
import json
import math
import os
from collections import Counter
from typing import Tuple, List

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage.registration import phase_cross_correlation

from utils import NumpyJsonEncoder, EIGHTHS, RED, GREEN, BLACK, BLUE
import line_detection

# point = (x, y)
# in cv2, y = 0 at the top
#         x = 0 at the left
# cv2 ndarray.shape = (height, width)
# pixel is ndarray[y, x]

# NOTE: cv2.line(...) correctly handles endpoints outside the image without
# raising exceptions, so extrapolation does NOT need boundary safety

# parameters for text
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2

FONT_ARGS = {'fontFace': FONT,
             'fontScale': FONT_SCALE,
             'thickness': FONT_THICKNESS}

TEXT_STRIP_HEIGHT = 1.8

# smallest rotation that will actually change the image
MIN_ROTATION = math.degrees(math.atan(2 / 5616))


def solve_line(point0, point1) -> Tuple[float, float]:
    """
    Solve y = ax + b from two points
    :param point0: (x, y)
    :param point1: (x, y)
    :return: a, b
    """
    a = (point1[1] - point0[1]) / (point1[0] - point0[0])
    b = point1[1] - a * point1[0]
    return a, b


def measure(line0):
    """
    Measure the length and direction of a line (CW from +x axis).
    Note: because in opencv y increases down the image, the angles
    are backwards from usual trigonometric convention.
    :param line0: (x0, y0, x1, y1) format returned by HoughLinesP
    :return: degrees, length, line (rounded to int)
    """
    dy = line0[3] - line0[1]
    dx = line0[2] - line0[0]
    angle = math.degrees(math.atan2(dy, dx))
    angle = normalize_angle(angle)
    # Pythagoras
    length = math.sqrt(dy * dy + dx * dx)
    line1 = [round(item) for item in line0]
    return angle, length, line1


def normalize_angle(angle):
    """
    Normalize the angle to the range -45° <= angle < 135° to simplify
    horizontal and vertical detection.
    :param angle:
    :return:
    """
    while angle < -45:
        angle += 180
    while angle >= 135:
        angle -= 180
    return angle


def binned_mode(values, bins=10, bin_range=None) -> float:
    """
    Make histogram bins and find the mode.
    :param values:
    :param bins: number of bins
    :param bin_range: (otherwise determined from values)
    :return: the mode
    """
    # TODO maybe implement the proper histogram mode formula
    # https://www.statology.org/histogram-mode/
    if bin_range:
        range0 = bin_range
    else:
        range0 = (min(values), max(values))
    counts, lefts = np.histogram(values, bins=bins, range=range0)
    i = counts.argmax()
    return (lefts[i] + lefts[i + 1]) / 2.0


def identify_image_sides(image_median, left_median, right_median,
                         very_left_median, very_right_median, config) -> Tuple[str, str]:
    """
    Identify left and right ends of image as 'light', 'dark', or 'unknown'
    :return: left and right 'light' or 'dark'
    """
    left_result = margin_median_result(left_median, image_median, config)
    right_result = margin_median_result(right_median, image_median, config)
    print(f'Left: {left_median} {left_result}')
    print(f'Right: {right_median} {right_result}')
    # fallback for narrow margins
    if (left_result == 'unknown') and (right_result == 'unknown'):
        left_result = margin_median_result(very_left_median, image_median, config)
        right_result = margin_median_result(very_right_median, image_median, config)
    # The binding is hard to detect so let's fudge it
    if left_result == 'unknown' and right_result == 'dark':
        left_result = 'light'
    elif left_result == 'dark' and right_result == 'unknown':
        right_result = 'light'
    elif left_result == 'unknown' and right_result == 'light':
        left_result = 'dark'
    elif left_result == 'light' and right_result == 'unknown':
        right_result = 'dark'
    print(f'Updated: left {left_result}, right {right_result}')
    return left_result, right_result


def get_edge_medians(image: np.ndarray, config) -> Tuple[int, int, int, int]:
    """
    Get medians for the whole image and the left and right parts.
    Note that the array has shape (vertical, horizontal)
    :param image: greyscale image
    :param config: config file object
    :return: overall, left, right
    """
    left = round(image.shape[1] * config['margin_detection']['left_right_test'])
    very_left = config['margin_detection']['fallback_side_test']
    left_median = round(np.median(image[:, :left]))
    very_left_median = round(np.median(image[:, :very_left]))
    right_median = round(np.median(image[:, - left:]))
    very_right_median = round(np.median(image[:, - very_left:]))
    return left_median, right_median, very_left_median, very_right_median


def margin_median_result(test_median, overall_median, config) -> str:
    """
    Determine the condition of the left and white sides (binding or black background)
    :param config: config file object
    :param test_median: median of the left or right edge
    :param overall_median: median of the whole image
    :return: 'light', 'dark', or 'unknown'
    """
    black_limit = overall_median * config['margin_detection']['black_limit']
    if test_median >= (overall_median + 3):
        result = 'light'
    elif test_median < black_limit:
        result = 'dark'
    else:
        result = 'unknown'
    return result


def rolling_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the window average over a series of values
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    :param values: list of values to average:
    :param window_size: size:
    :return: array with the same size as the input array
    """
    slider = np.ones(window_size) / window_size
    return np.convolve(values, slider, 'same')


def smoothed_vertical_medians(image: np.ndarray, config) -> np.ndarray:
    """
    Calculate the smoothed median of each vertical line
    :param image: greyscale image
    :param config: config file object
    :return: array with the image width
    """
    medians = np.median(image, axis=0)
    window = config['margin_detection']['smoothing_window']
    return rolling_average(medians, window)


def smoothed_horizontal_medians(image: np.ndarray, config):
    """
    Calculate the smoothed median of each horizontal line
    :param image: greyscale image
    :param config: config file object
    :return: array with the image height
    """
    medians = np.median(image, axis=1)
    window = config['margin_detection']['smoothing_window']
    return rolling_average(medians, window)


def get_top_bottom_cutoffs(image: np.ndarray, config) -> Tuple[int, int]:
    """
    Scan horizontal lines to find the top and bottom margins
    :param image: greyscale image
    :param config: config file object
    :return: two y-values
    """
    y_max = image.shape[0] - 1
    image_median = np.median(image)
    black_limit = image_median * config['margin_detection']['black_limit']
    y_fraction = config['margin_detection']['top_bottom_limit']
    h_medians = smoothed_horizontal_medians(image, config)
    top_cut = 0
    for y in range(0, round(image.shape[0] * y_fraction)):
        if h_medians[y] > black_limit:
            top_cut = y
            break

    bottom_cut = y_max
    for y in range(y_max, round(y_max * (1 - y_fraction)), -1):
        if h_medians[y] > black_limit:
            bottom_cut = y
            break
    return top_cut, bottom_cut


def get_left_cutoff(medians, edge_type, overall_median, config) -> int:
    """
    Find the maximum X value for the left margin
    :param medians: smoothed medians of vertical lines
    :param edge_type: 'light', 'dark', or 'unknown'
    :param overall_median: median of the whole image
    :param config: config file object
    :return:
    """
    black_limit = overall_median * config['margin_detection']['black_limit']
    max_left = round(len(medians) * config['margin_detection']['left_right_limit'])
    cutoff = 0
    if edge_type == 'light':
        for x in range(max_left, 0, -1):
            if medians[x] > overall_median + 5:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(0, max_left):
            if medians[x] > black_limit:
                cutoff = x
                break
    return cutoff


def get_right_cutoff(medians, edge_type, overall_median, config) -> int:
    """
    Find the minimum X value for the right margin
    :param medians: smoothed medians of vertical lines
    :param edge_type: 'light', 'dark', or 'unknown'
    :param overall_median: median of the whole image
    :param config: config file object
    :return:
    """
    black_limit = overall_median * config['margin_detection']['black_limit']
    min_right = round(len(medians) * (1 - config['margin_detection']['left_right_limit']))
    cutoff = len(medians) - 1
    if edge_type == 'light':
        for x in range(min_right, len(medians)):
            if medians[x] > overall_median + 5:
                cutoff = x
                break
    elif edge_type == 'dark':
        for x in range(len(medians) - 1, min_right, -1):
            if medians[x] > black_limit:
                cutoff = x
                break
    return cutoff


def rotate_correct(image: np.ndarray, config: dict, output_dir: str, output_prefix: str,
                   sheet_info: dict) -> np.ndarray:
    deviation = config['rotation']['deviation']
    mode_bins = config['rotation']['mode_bins']
    ignore_sides_fraction = config['rotation']['ignore_sides_fraction']
    ignore_top_fraction = config['rotation']['ignore_top_fraction']
    ignore_bottom_fraction = config['rotation']['ignore_bottom_fraction']
    left_limit = round(image.shape[1] * ignore_sides_fraction)
    right_limit = round(image.shape[1] * (1 - ignore_sides_fraction))
    top_limit = round(image.shape[0] * ignore_top_fraction)
    bottom_limit = round(image.shape[0] * (1- ignore_bottom_fraction))
    # TODO investigate parameters
    lines = line_detection.detect_lines_within(image, left_limit, right_limit, top_limit, bottom_limit)
    horizontal_lines = line_detection.keep_horizontal(lines, deviation)
    # debug: save the hlines image
    if config['debug']:
        debug_image_path = os.path.join(output_dir, f'{output_prefix}_0_hlines.png')
        hline_image = generate_hlines_image(image, horizontal_lines)
        print(f'Writing {debug_image_path}')
        cv2.imwrite(debug_image_path, hline_image)
    # find the binned mode of angles and rotate
    angles = [line.angle for line in horizontal_lines]
    angle_octiles = np.quantile(angles, EIGHTHS)
    rotation = binned_mode(angles, bins=mode_bins, bin_range=(-deviation, deviation))
    rotation = round(rotation, 3)
    if abs(rotation) > MIN_ROTATION:
        print(f'Rotating {rotation:.3f} degrees')
        # ndimage angle in degrees
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
        output = ndimage.rotate(image, angle=rotation, reshape=False)
        sheet_info['rotation'] = {'degrees': rotation,
                                  'applied': True,
                                  'octiles': angle_octiles}
    else:
        print(f'Ignoring {rotation:.3f} degrees')
        output = image
        sheet_info['rotation'] = {'degrees': rotation,
                                  'applied': False,
                                  'octiles': angle_octiles}
    return output


def erase_margins(image: np.ndarray, config):
    # set some values
    image_median = round(np.median(image))
    x_max = image.shape[1] - 1
    y_max = image.shape[0] - 1
    print(f'Median: {image_median}')

    # scan horizontal lines
    top_cutoff, bottom_cutoff = get_top_bottom_cutoffs(image, config)
    modified_image = image.copy()
    # erase the top and bottom margins
    for y in range(0, top_cutoff):
        cv2.line(modified_image, (0, y), (x_max, y), image_median, 1)
    for y in range(bottom_cutoff, y_max):
        cv2.line(modified_image, (0, y), (x_max, y), image_median, 1)

    # scan vertical lines from the MODIFIED image
    # left_median, right_median = get_edge_medians(modified_image, config)
    # try using vertical lines from the ORIGINAL
    left_median, right_median, very_left_median, very_right_median = get_edge_medians(image, config)
    print(f'Medians: left {left_median}, very left {very_left_median}, '
          f'right {right_median}, very right {very_right_median}')
    side_types = identify_image_sides(image_median, left_median, right_median,
                                      very_left_median, very_right_median, config)
    v_medians = smoothed_vertical_medians(modified_image, config)
    left_cutoff = get_left_cutoff(v_medians, side_types[0], image_median, config)
    right_cutoff = get_right_cutoff(v_medians, side_types[1], image_median, config)
    found_width = right_cutoff - left_cutoff
    print(f'Detected cutoffs: left {left_cutoff}, right {right_cutoff}, width {found_width}')
    min_width = config['margin_detection']['min_width']
    if found_width < min_width:
        if side_types[0] == 'light':
            left_cutoff -= found_width - min_width
        elif side_types[1] == 'light':
            right_cutoff += found_width - min_width
        else:
            adjustment = (min_width - found_width) // 2
            left_cutoff -= adjustment
            right_cutoff += adjustment
        print(f'Modified cutoffs: left {left_cutoff}, right {right_cutoff}, width {right_cutoff - left_cutoff}')
    # and erase the left and right margins
    for x in range(0, left_cutoff):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    for x in range(right_cutoff, image.shape[1]):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    print(f'Cutoffs: left {left_cutoff}, right {right_cutoff}, top {top_cutoff}, bottom {bottom_cutoff}')
    return modified_image, (left_cutoff, right_cutoff, top_cutoff, bottom_cutoff)


def get_lined_image_for_registration(image: np.ndarray, config: dict, cutoffs: tuple) -> np.ndarray:
    """
    Generate a black on white image of horizontal and vertical lines (for
    subsequent registration by phase cross correlation).
    :param image: input image
    :param config: configuration parameters
    :param cutoffs: tuple of horizontal and vertical lines coordinates
    :return: white image with black lines for registration
    """
    lines = line_detection.detect_lines_within(image, *cutoffs)
    h_tolerance = config['registration']['horizontal']['deviation']
    v_tolerance = config['registration']['vertical']['deviation']
    h_and_v_lines = line_detection.keep_horizontal_and_vertical(lines, h_tolerance, v_tolerance)
    # create white image
    new_image = np.full(image.shape, 255, dtype="uint8")
    # superimpose black lines
    for line in h_and_v_lines:
        cv2.line(new_image, (line.x0, line.y0), (line.x1, line.y1), (0,), 3)
    return new_image


def get_registration_width(registration_image: np.ndarray, sheet_info: dict,
                           percentile: float, cutoffs: Tuple[int, int, int, int],
                           max_header_width: int) -> Tuple[int, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Determine the width of the header box detected from the muster image.
    :param registration_image: input image (generated by line segment detection)
    :param sheet_info:
    :param percentile: percentile of column values for determining left and right edges of header
    :param cutoffs left, right, top, bottom of search box for header
    :param max_header_width: maximum allowed width of header box
    :return: the width of the header box (to be used for rescaling)
    """
    # sum the image vertically and count white pixels
    column_total = registration_image.sum(axis=0)
    column_counts_white = np.floor_divide(column_total, 255)
    # subtract to get nbr of black pixels (dtype is now int64) in each column
    column_counts_black = registration_image.shape[0] - column_counts_white
    # for threshold calculation, only use the area within the box
    column_counts_usable = column_counts_black[cutoffs[0]:cutoffs[1]]
    threshold = np.percentile(column_counts_usable, percentile)
    print(f'Column threshold {threshold}, min {column_counts_black.min()}, max {column_counts_black.max()}')
    # https://stackoverflow.com/questions/76786670/what-is-the-difference-between-threshold-and-prominence-in-scipy-find-peaks
    strong_columns = signal.find_peaks(column_counts_black, prominence=threshold)[0]
    # np.where equivalent returns a singleton tuple here
    # https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    strong_left, strong_right, box_width, edge_log = determine_header_width(strong_columns, max_header_width)
    # IMPORTANT, PENDING
    # TODO use max_header_width, probably to suppress the LH edge
    octiles = np.quantile(column_counts_black, EIGHTHS)
    nonzero = np.asarray(column_counts_black > 0).nonzero()[0]
    rescaling = {'octiles': octiles,
                 'strong_edges': [[strong_left, column_counts_black[strong_left]],
                                  [strong_right, column_counts_black[strong_right]]],
                 'percentile': percentile,
                 'threshold': threshold,
                 'peaks': strong_columns,
                 'edge_log': edge_log,
                 'min_black': column_counts_black.min(),
                 'max_black': column_counts_black.max(),
                 'nonzero_range': [[nonzero[0], column_counts_black[nonzero[0]]],
                                   [nonzero[-1], column_counts_black[nonzero[-1]]]],
                 'found_box_width': box_width}
    sheet_info['rescaling'] = rescaling
    return box_width, column_counts_black, (strong_left, strong_right), octiles


def determine_header_width(strong_columns: np.ndarray, max_header_width: int) -> Tuple[int, int, int, list]:
    print('Strong columns:', strong_columns)
    left_index = 0
    strong_left, strong_right = strong_columns[left_index], strong_columns[-1]
    box_width = strong_right - strong_left
    print(f'Using: {strong_left}, {strong_right} -> {box_width} [{max_header_width}]')
    log = [(strong_left, strong_right, box_width, max_header_width)]
    while box_width > max_header_width:
        left_index += 1
        strong_left = strong_columns[left_index]
        box_width = strong_right - strong_left
        log.append((strong_left, strong_right, box_width, max_header_width))
        print(f'Using: {strong_left}, {strong_right} -> {box_width} [{max_header_width}]')
    return strong_left, strong_right, box_width, log


def adjust_muster_to_template(image: np.ndarray, template: np.ndarray,
                              template_form_width: int, config: dict,
                              output_dir: str, output_prefix: str,
                              sheet_info: dict) -> np.ndarray:
    """
    Rotate the image; erase the margins; detect H and V lines and shift to match the template
    :param image: muster sheet
    :param template: template image
    :param template_form_width: pixel width of form
    :param config: config file object
    :param output_dir: used for debugging
    :param output_prefix: used for debugging
    :param sheet_info: cumulative log of sheet processing
    """
    modified_image = rotate_correct(image, config, output_dir, output_prefix, sheet_info)
    modified_image, cutoffs = erase_margins(modified_image, config)
    # cutoffs (left, right, top, bottom)
    adjusted_cutoffs = (cutoffs[0] + config['registration']['cutoff_adjustments']['left'],
                        cutoffs[1] - config['registration']['cutoff_adjustments']['right'],
                        cutoffs[2] + config['registration']['cutoff_adjustments']['top'],
                        cutoffs[2] + config['registration']['cutoff_adjustments']['top'] +
                        config['registration']['cutoff_adjustments']['top_band'],
                        )
    print(f'Inner cutoffs: {adjusted_cutoffs}')
    sheet_info['cutoffs'] = {'outer': cutoffs, 'inner': adjusted_cutoffs}
    # Detect H & V lines for the dark boxes at the top of the form
    registration_image = get_lined_image_for_registration(modified_image, config, adjusted_cutoffs)
    percentile = config['registration']['percentile']
    max_header_factor = config['registration']['max_header_factor']
    max_header_width = round(template_form_width * max_header_factor)
    muster_box_width, column_counts, header_edges, registration_octiles = get_registration_width(registration_image,
                                                                                                 sheet_info, percentile,
                                                                                                 adjusted_cutoffs,
                                                                                                 max_header_width)
    rescale_factor = template_form_width / muster_box_width
    sheet_info['rescaling']['rescale'] = rescale_factor
    print(f'Template box width {template_form_width} vs muster {muster_box_width} -> rescale {rescale_factor:.3f}')
    # rescale muster image but only horizontally to match the template
    print(f'Before clipped_zoom {modified_image.shape}')
    rescaled_image = rescale_horizontally(modified_image, rescale_factor)
    rescaled_registration_image = rescale_horizontally(registration_image, rescale_factor)
    print(f'After clipped_zoom {rescaled_image.shape}')
    sheet_info['shape']['input'] = image.shape
    sheet_info['shape']['modified'] = modified_image.shape
    sheet_info['shape']['rescaled'] = rescaled_image.shape

    # PCC works much better without normalization
    shift, error, diffphase = phase_cross_correlation(template, rescaled_registration_image, normalization=None)
    sheet_info['pcc'] = {'shift': shift, 'error': error, 'diffphase': diffphase}
    # pass this shift value as argument to scipy.ndimage.shift
    print('Registration:', shift, error, diffphase)
    # Shift vector (in pixels) required to register moving_image with reference_image.
    # Axis ordering is consistent with the axis order of the input array.
    # https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
    fully_adjusted_image = ndimage.shift(rescaled_image, shift)

    if config['debug']:
        debug_image_path = os.path.join(output_dir, f'{output_prefix}_1_cutoffs.png')
        debug_image = mark_cutoffs(modified_image, cutoffs, adjusted_cutoffs, registration_image,
                                   column_counts, header_edges, registration_octiles)
        print(f'Writing {debug_image_path}')
        cv2.imwrite(debug_image_path, debug_image)
        columns_path = os.path.join(output_dir, f'{output_prefix}_1_columns.json')
        print(f'Writing {columns_path}')
        with open(columns_path, 'w') as f:
            json.dump(column_counts, f, cls=NumpyJsonEncoder)

    return fully_adjusted_image


def superimpose(shifted_image: np.ndarray, form: np.ndarray) -> np.ndarray:
    """
    Superimpose the form in blue on the image
    :param shifted_image: grayscale image
    :param form: template to add
    :return: merged RGB image
    """
    if shifted_image.shape != form.shape:
        raise ValueError(f'Not matched: {shifted_image.shape} {form.shape}')
    output = cv2.cvtColor(shifted_image, cv2.COLOR_GRAY2RGB)
    output[np.where(form == 0)] = BLUE
    return output


def mark_cutoffs(image: np.ndarray, outer: tuple, inner: tuple,
                 registration_image: np.ndarray,
                 column_counts: np.ndarray, header_edges: Tuple[int, int],
                 registration_octiles: np.ndarray) -> np.ndarray:
    """
    Superimpose the form in blue on the image
    :param image: grayscale image
    :param outer: (left, right, top, bottom)
    :param inner: (left, right, top, bottom)
    :param registration_image: H & V lines detected
    :param column_counts: darkness of columns within general header zone
    :param header_edges: left and right edges of found header for rescaling
    :param registration_octiles: octiles of column values for debugging
    :return: merged RGB image
    """
    left_outer, right_outer, top_outer, bottom_outer = outer
    left_inner, right_inner, top_inner, bottom_inner = inner
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print(f'Debugging mark_cutoffs: {output.shape}')
    # mark the margins (erased outside these)
    cv2.line(output, (left_outer, top_outer), (right_outer, top_outer), RED, 3)
    cv2.line(output, (right_outer, top_outer), (right_outer, bottom_outer), RED, 3)
    cv2.line(output, (right_outer, bottom_outer), (left_outer, bottom_outer), RED, 3)
    cv2.line(output, (left_outer, bottom_outer), (left_outer, top_outer), RED, 3)
    # mark the adjusted cutoffs (area used for header registration)
    cv2.line(output, (left_inner, top_inner), (right_inner, top_inner), RED, 3)
    cv2.line(output, (right_inner, top_inner), (right_inner, bottom_inner), RED, 3)
    cv2.line(output, (right_inner, bottom_inner), (left_inner, bottom_inner), RED, 3)
    cv2.line(output, (left_inner, bottom_inner), (left_inner, top_inner), RED, 3)
    # copy the H & V lines detected
    output[np.where(registration_image == 0)] = BLUE
    # draw green lines above the detected box edges
    cv2.line(output, (header_edges[0], top_outer), (header_edges[0], top_inner), GREEN, 3)
    cv2.line(output, (header_edges[1], top_outer), (header_edges[1], top_inner), GREEN, 3)
    # mark the column counts used for determining the header width
    scale_v = (top_outer - 1) / column_counts.max()
    # plot the registration values across the top
    for x, count in enumerate(column_counts):
        y = round(scale_v * count)
        cv2.line(output, (x, 0), (x, y), GREEN, 1)
    # mark the registration octiles on the left
    scale_h = (left_outer - 5) / (registration_octiles.shape[0] + 1)
    for i, y in enumerate(registration_octiles):
        x1 = round(scale_h * (i + 1))
        y1 = round(scale_v * y)
        cv2.line(output, (0, y1), (x1, y1), GREEN, 1)
    return output


def generate_hlines_image(image: np.ndarray, hlines: List[line_detection.Line]) -> np.ndarray:
    """
    Superimpose the near-horizontal lines on the image
    :param image: grayscale image
    :param hlines: list of measured lines
    :return: merged RGB image
    """
    # angle, length, [x0, y0, x1, y1]
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for line in hlines:
        p0 = (line.x0, line.y0)
        p1 = (line.x1, line.y1)
        cv2.line(output, p0, p1, BLUE, 1)
    return output


def generate_image_for_correction(image: np.ndarray, cell_spec, ml_output: str, workflow_name: str):
    """
    Generate the image for Zooniverse phase 2.
    :param image: registered image
    :param cell_spec: cell specification
    :param ml_output: HTR output
    :param workflow_name: name of workflow
    """
    # calculate sizes of texts
    # https://lindevs.com/get-text-size-using-opencv
    workflow_text = f'Field: {workflow_name}'
    (wf_width, wf_height), wf_baseline = cv2.getTextSize(workflow_text, **FONT_ARGS)
    wf_full_height = wf_height + wf_baseline
    (htr_width, htr_height), htr_baseline = cv2.getTextSize(ml_output, **FONT_ARGS)
    htr_full_height = htr_height + htr_baseline
    htr_width += 15
    required_width = max(wf_width, htr_width) + 25
    extra_height_top = round(wf_full_height * TEXT_STRIP_HEIGHT)
    extra_height_bottom = round(htr_full_height * TEXT_STRIP_HEIGHT)
    # crop image to outer bounds
    outer_left, outer_right = cell_spec['expanded']['left'], cell_spec['expanded']['right']
    outer_top, outer_bottom = cell_spec['expanded']['top'], cell_spec['expanded']['bottom']
    side_expansion = max(0, required_width - (outer_right - outer_left))
    outer_left -= side_expansion // 2
    outer_right += side_expansion // 2
    extracted_image = image[outer_top: outer_bottom, outer_left: outer_right]
    # calculate red box WITHIN cropped image after adding workflow to the top
    inner_left, inner_right = cell_spec['htr']['left'] - outer_left, cell_spec['htr']['right'] - outer_left
    inner_top = cell_spec['htr']['top'] - outer_top + extra_height_top
    inner_bottom = cell_spec['htr']['bottom'] - outer_top + extra_height_top
    # add white strips at the top and bottom to accommodate the texts
    output_width = extracted_image.shape[1]
    top_strip = np.full((extra_height_top, output_width), 255, dtype=extracted_image.dtype)
    bottom_strip = np.full((extra_height_bottom, output_width), 255, dtype=extracted_image.dtype)
    extracted_image = np.vstack((top_strip, extracted_image, bottom_strip))
    # add the red box
    output_image = cv2.cvtColor(extracted_image, cv2.COLOR_GRAY2RGB)
    cv2.line(output_image, (inner_left, inner_top), (inner_left, inner_bottom), RED, 1)
    cv2.line(output_image, (inner_right, inner_top), (inner_right, inner_bottom), RED, 1)
    cv2.line(output_image, (inner_left, inner_top), (inner_right, inner_top), RED, 1)
    cv2.line(output_image, (inner_left, inner_bottom), (inner_right, inner_bottom), RED, 1)
    # add the text, centered horizontally
    wf_left = (output_width - wf_width) // 2
    wf_bottom = wf_height + (extra_height_top - wf_height) // 2
    htr_left = (output_width - htr_width) // 2
    htr_bottom = output_image.shape[0] - (extra_height_bottom - htr_height) // 2
    cv2.putText(img=output_image, text=workflow_text, org=(wf_left, wf_bottom), color=BLACK, **FONT_ARGS)
    cv2.putText(img=output_image, text=ml_output, org=(htr_left, htr_bottom), color=BLACK, **FONT_ARGS)
    return output_image


def resize_image(source: np.ndarray, to_match: np.ndarray) -> np.ndarray:
    """
    Change the size of the matrix WITHOUT rescaling.
    """
    # typically (3744, 5616)
    if source.shape == to_match.shape:
        return source
    # TODO
    output = source
    target_height, target_width = to_match.shape
    source_height, source_width = source.shape
    # Crop the image by deleting from the ends;
    # expand the image by duplicating the right or bottom line
    if source_height > target_height:
        i = round((source_height - target_height) / 2)
        j = i + target_height
        output = output[i:j, :]
    elif source_height < target_height:
        output = dup_rows(output, target_height - source_height)
    if source_width > target_width:
        i = round((source_width - target_width) / 2)
        j = i + target_width
        output = output[:, i:j]
    elif source_width < target_width:
        output = dup_cols(output, target_width - source_width)
    print(f'Resized to {output.shape}')
    return output


# https://stackoverflow.com/questions/43870647/how-to-duplicate-a-row-or-column-in-a-numpy-array
def dup_rows(a, number=1):
    index = a.shape[0] - 1
    return np.insert(a, [index + 1] * number, a[index], axis=0)


def dup_cols(a, number=1):
    index = a.shape[1] - 1
    return np.insert(a, [index + 1] * number, a[:, [index]], axis=1)


def rescale_horizontally(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Rescale horizontally but return ndarray with the same shape (crop or fill as needed)
    :param image: grayscale image
    :param scale: scale factor
    :return: resized image.
    """
    if scale == 1.0:
        return image
    # y, x
    zoom_tuple = (1, scale)
    original_width = image.shape[1]
    modified_image = ndimage.zoom(image, zoom_tuple)
    if modified_image.shape[1] > original_width:
        modified_image = modified_image[:, :original_width]
    elif modified_image.shape[1] < original_width:
        # https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array
        plus_columns = original_width - modified_image.shape[1]
        height = modified_image.shape[0]
        extension = np.full(shape=(height, plus_columns), fill_value=np.median(modified_image),
                            dtype=modified_image.dtype)
        modified_image = np.hstack((modified_image, extension))
    return modified_image


def find_hlines(image: np.ndarray, config: dict, y_min: int, y_max: int,
                y_ref_line: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the slightly smoothed mean pixel value of the middle half of the image to find
    row boundaries (which are brighter than the content).
    :param image: grayscale image
    :param config: config parameters
    :param y_min: minimum y-coordinate (derived from template)
    :param y_max: maximum y-coordinate (from template)
    :param y_ref_line: top of top row according to the template
    :return: row cutoffs
    """
    left_cut = image.shape[1] // 4
    middle_half = image[:, left_cut: -left_cut]
    horizontal_means = np.mean(middle_half, axis=1)
    window_size = config['row_detection']['smoothing_window']
    window = np.ones(window_size) / window_size
    horizontal_means = np.convolve(horizontal_means, window, 'same')
    min_distance = config['row_detection']['min_distance']
    max_distance = config['row_detection']['max_distance']
    peaks = signal.find_peaks(horizontal_means, distance=min_distance)[0]
    peaks = peaks[np.where(peaks > y_min)]
    peaks = peaks[np.where(peaks < y_max)]
    peaks = interpolate_hlines(peaks, max_distance)
    if abs(y_ref_line - peaks[0]) < min_distance:
        peaks[0] = min(y_ref_line, peaks[0])
    else:
        peaks = np.insert(peaks, 0, y_ref_line)
    return peaks, horizontal_means


def interpolate_hlines(peaks: np.ndarray, max_distance: int) -> np.ndarray:
    """
    Interpolate additional row boundaries where the gap exceeds max_distance.
    :param peaks: row cutoffs before interpolation
    :param max_distance: maximum distance between peaks
    :return: row boundaries with interpolation
    """
    differences = peaks[1:] - peaks[:-1]
    insertion_points = []
    insertions = []
    for i, difference in enumerate(differences):
        if difference > max_distance:
            nbr = difference // max_distance + 1
            for j in range(1, nbr):
                insertion_points.append(i + 1)
                insertions.append((peaks[i + 1] + peaks[i]) * j // nbr)
    return np.insert(peaks, insertion_points, insertions)

# library file

import math
from collections import Counter
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

# point = (x, y)
# in cv2, y = 0 at the top
#         x = 0 at the left
# cv2 ndarray.shape = (height, width)

# NOTE: cv2.line(...) correctly handles endpoints outside the image without
# raising exceptions, so extrapolation does NOT need boundary safety

# the cv2 color scheme is (blue, green, red)
RED = (0, 0, 255)
GREEN = (0, 128, 0)
BLUE = (255, 0, 0)


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


def extrapolate_vertical(point0, point1, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Extrapolate to the top and bottom edges.
    :param point0:
    :param point1:
    :param image:
    :return: top_point, bottom_point, horizontal_difference
    """
    max_x, max_y = get_maxima(image)
    x0, y0 = point0
    x1, y1 = point1

    # special case: vertical line has infinite slope
    if x0 == x1:
        return (x0, 0), (x0, max_y), 0

    a, b = solve_line(point0, point1)
    top = (round(-b / a), 0)
    bottom = (round((max_y - b) / a), max_y)
    x_diff = abs(bottom[0] - top[0])
    return top, bottom, x_diff


def extrapolate_horizontal(point0, point1, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Extrapolate to the left and right edges.
    :param point0:
    :param point1:
    :param image:
    :return: left_point, right_point, vertical_difference
    """
    max_x, max_y = get_maxima(image)
    a, b = solve_line(point0, point1)

    left = (0, round(b))
    right = (max_x, round(a * max_x + b))
    y_diff = abs(right[1] - left[1])

    return left, right, y_diff


def measure(line0):
    """
    Measure the length and direction of a line (CW from +x axis).
    Note: because in opencv y increases down the image, the angles
    are backwards from usual trigonometric convention.
    :param line0: (x0, y0, x1, y1) format returned by HoughLinesP
    :return: degrees, length, line (repeated in same format)
    """
    dy = line0[3] - line0[1]
    dx = line0[2] - line0[0]
    angle = math.degrees(math.atan2(dy, dx))
    # Pythagoras
    length = math.sqrt(dy * dy + dx * dx)
    return angle, length, line0


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


def is_nearly_horizontal(angle, tolerance):
    norm_angle = normalize_angle(angle)
    return (0 - tolerance) < norm_angle < (0 + tolerance)


def is_nearly_vertical(angle, tolerance):
    norm_angle = normalize_angle(angle)
    return (90 - tolerance) < norm_angle < (90 + tolerance)


def get_maxima(image: np.ndarray) -> Tuple[int, int]:
    return image.shape[0] - 1, image.shape[1] - 1


def binned_mode(values, bins=10, range=None) -> float:
    """
    Make histogram bins and find the mode.
    :param values:
    :param bins: number of bins
    :param range: (otherwise determined from values)
    :return: the mode
    """
    # TODO maybe implement the proper histogram mode formula
    # https://www.statology.org/histogram-mode/
    if range:
        range0 = range
    else:
        range0 = (min(values), max(values))
    counts, lefts = np.histogram(values, bins=bins, range=range0)
    i = counts.argmax()
    return (lefts[i] + lefts[i + 1]) / 2.0


def most_frequent_values(values, interval, end, count):
    """
    Bin the values histogram-style and return the middles and
    counts of the most frequent bins
    :param values: sequence of values
    :param interval: bin size
    :param end: high value of highest bin
    :param count: number of values to return
    :return:
    """
    bins = range(0, end, interval)
    counts, edges = np.histogram(values, bins=bins)
    middles = [round((edges[i] + edges[i + 1]) / 2) for i in range(len(edges) - 1)]
    d = {middles[i]: counts[i] for i in range(len(counts)) if (counts[i] > 0)}
    c = Counter(d)
    return c.most_common(count)


def identify_image_sides(image_median, left_median, right_median, config) -> Tuple[str, str]:
    """
    Identify left and right ends of image as 'light', 'dark', or 'unknown'
    :return: left and right 'light' or 'dark'
    """
    left_result = margin_median_result(left_median, image_median, config)
    right_result = margin_median_result(right_median, image_median, config)
    print(f'Left: {left_median} {left_result}')
    print(f'Right: {right_median} {right_result}')
    # The binding is hard to detect so let's fudge it
    if left_result == 'unknown' and right_result == 'dark':
        left_result = 'light'
    elif left_result == 'dark' and right_result == 'unknown':
        right_result = 'light'
    print(f'Updated: left {left_result}, right {right_result}')
    return left_result, right_result


def get_edge_medians(image: np.ndarray, config) -> Tuple[int, int]:
    """
    Get medians for the whole image and the left and right parts.
    Note that the array has shape (vertical, horizontal)
    :param image: greyscale image
    :param config: config file object
    :return: overall, left, right
    """
    left = round(image.shape[1] * config['margin_detection']['left_right_test'])
    left_median = round(np.median(image[:, :left]))
    right_median = round(np.median(image[:, -left:]))
    return left_median, right_median


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
    return np.convolve(values, np.ones(window_size), 'same') / window_size


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


def generate_test_image(image, horizontal_lines, vertical_lines, debug):
    if not debug:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for measured_line in horizontal_lines + vertical_lines:
        line = measured_line[2]
        cv2.line(image_rgb, (line[0], line[1]), (line[2], line[3]), BLUE, 1)
    return image_rgb


def rotate_correct(image: np.ndarray, config: dict) -> np.ndarray:
    canny_min = config['rotation']['canny_min']
    canny_max = config['rotation']['canny_max']
    canny_aperture = config['rotation']['canny_aperture']
    hough_threshold = config['rotation']['hough_threshold']
    hough_min_length = config['rotation']['hough_min_length']
    hough_max_gap = config['rotation']['hough_max_gap']
    deviation = config['rotation']['deviation']
    dst = cv2.Canny(image, canny_min, canny_max, None, canny_aperture, True)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, hough_threshold, None, hough_min_length, hough_max_gap)
    measured_lines = [measure(line[0]) for line in lines.tolist()]
    h_lines = [d for d in measured_lines if is_nearly_horizontal(d[0], deviation)]
    angles = [d[0] for d in h_lines]
    rotation = binned_mode(angles, bins=21, range=(-deviation, deviation))
    return ndimage.rotate(image, angle=rotation, reshape=False)


def erase_margins(image: np.ndarray, config):
    # set some values
    image_median = round(np.median(image))
    x_max = image.shape[1] - 1
    y_max = image.shape[0] - 1
    print(f'Shape: {image.shape}')
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
    left_median, right_median = get_edge_medians(modified_image, config)
    print(f'Side medians: left {left_median}, right {right_median}')
    side_types = identify_image_sides(image_median, left_median, right_median, config)
    v_medians = smoothed_vertical_medians(modified_image, config)
    left_cutoff = get_left_cutoff(v_medians, side_types[0], image_median, config)
    right_cutoff = get_right_cutoff(v_medians, side_types[1], image_median, config)
    # and erase the left and right margins
    for x in range(0, left_cutoff + 1):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    for x in range(right_cutoff, image.shape[1]):
        cv2.line(modified_image, (x, 0), (x, y_max), image_median, 1)
    print(f'Cutoffs: top {top_cutoff}, bottom {bottom_cutoff}, left {left_cutoff}, right {right_cutoff}')
    return modified_image, (left_cutoff, right_cutoff, top_cutoff, bottom_cutoff)


def keep_horizontal_or_vertical(measured_line, tolerance, cutoffs=None):
    # cutoffs: left, right, top, bottom
    angle = measured_line[0]
    # ML: degrees 90 up, length,  line (x0, y0, x1, y1)
    x = (measured_line[2][0] + measured_line[2][2]) / 2
    y = (measured_line[2][1] + measured_line[2][3]) / 2
    left = min(measured_line[2][0], measured_line[2][2])
    right = max(measured_line[2][0], measured_line[2][2])
    top = min(measured_line[2][1], measured_line[2][3])
    bottom = max(measured_line[2][1], measured_line[2][3])
    if not cutoffs:
        result = is_nearly_vertical(angle, tolerance) or is_nearly_horizontal(angle, tolerance)
    elif is_nearly_vertical(angle, tolerance) and (cutoffs[0] < x < cutoffs[1]) and (top > cutoffs[2]) and (
            bottom < cutoffs[3]):
        result = True
    elif is_nearly_horizontal(angle, tolerance) and (cutoffs[2] < y < cutoffs[3]) and (left > cutoffs[0]) and (
            right < cutoffs[1]):
        result = True
    else:
        result = False
    return result


def get_lines_for_registration(image: np.ndarray, config: dict, cutoffs: tuple) -> np.ndarray:
    canny_min = config['registration']['canny_min']
    canny_max = config['registration']['canny_max']
    canny_aperture = config['registration']['canny_aperture']
    hough_threshold = config['registration']['hough_threshold']
    hough_min_length = config['registration']['hough_min_length']
    hough_max_gap = config['registration']['hough_max_gap']
    deviation = config['registration']['deviation']
    dst = cv2.Canny(image, canny_min, canny_max, None, canny_aperture, True)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, hough_threshold, None, hough_min_length, hough_max_gap)
    measured_lines = [measure(line[0]) for line in lines.tolist()]
    keep_lines = [line for line in measured_lines if keep_horizontal_or_vertical(line, deviation, cutoffs)]
    new_image = np.full(image.shape, 255, dtype="uint8")
    for line in keep_lines:
        cv2.line(new_image, (line[2][0], line[2][1]), (line[2][2], line[2][3]), (0,), 3)
    return new_image


def adjust_muster_to_template(image: np.ndarray, template: np.ndarray, config: dict) -> np.ndarray:
    """
    Rotate the image; erase the margins; detect H and V lines and shift to match the template
    :param image: muster sheet
    :param template: template image
    :param config: config file object
    """
    modified_image = rotate_correct(image, config)
    modified_image, cutoffs = erase_margins(modified_image, config)
    # cutoffs (left, right, top, bottom)
    adjusted_cutoffs = (cutoffs[0] + config['registration']['cutoff_adjustments']['left'],
                        cutoffs[1] - config['registration']['cutoff_adjustments']['right'],
                        cutoffs[2] + config['registration']['cutoff_adjustments']['top'],
                        cutoffs[3] - config['registration']['cutoff_adjustments']['bottom'])
    print(adjusted_cutoffs)
    registration_image = get_lines_for_registration(modified_image, config, adjusted_cutoffs)
    # works much better without normalization
    shift, error, diffphase = phase_cross_correlation(template, registration_image, normalization=None)
    # pass this shift value as argument to scipy.ndimage.shift
    print('Registration:', shift, error, diffphase)
    # Shift vector (in pixels) required to register moving_image with reference_image.
    # Axis ordering is consistent with the axis order of the input array.
    # https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation
    shifted_image = ndimage.shift(modified_image, shift)
    return shifted_image


def superimpose(shifted_image: np.ndarray, form: np.ndarray) -> np.ndarray:
    if shifted_image.shape != form.shape:
        raise ValueError(f'Not matched: {shifted_image.shape} {form.shape}')
    output = cv2.cvtColor(shifted_image, cv2.COLOR_GRAY2RGB)
    output[np.where(form == 0)] = BLUE
    return output


def resize_image(source: np.ndarray, to_match: np.ndarray) -> np.ndarray:
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
    return output


# https://stackoverflow.com/questions/43870647/how-to-duplicate-a-row-or-column-in-a-numpy-array
def dup_rows(a, number=1):
    index = a.shape[0] - 1
    return np.insert(a, [index + 1] * number, a[index], axis=0)


def dup_cols(a, number=1):
    index = a.shape[1] - 1
    return np.insert(a, [index + 1] * number, a[:, [index]], axis=1)

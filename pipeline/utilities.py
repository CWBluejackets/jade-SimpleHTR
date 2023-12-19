# library file
import json
import logging
import os.path
import re
from typing import Tuple

import cv2

# used to extract vessel name and page sequence from JPG filename
FILENAME_PATTERN = re.compile(r'\d+_(\S+)_(\d+)\.\w+')


def get_vessel_and_page(path: str) -> Tuple[str, int]:
    """
    Extract the vessel name and page sequence from the image filename
    :return: vessel, vessel_page
    """
    filename = os.path.basename(path)
    match = FILENAME_PATTERN.fullmatch(filename)
    if match:
        result = (match.group(1), int(match.group(2)))
    else:
        result = (None, None)
    return result


def write_control_file(input_path: str, output_directory: str, data: dict):
    input_name = os.path.basename(input_path)
    stem = os.path.splitext(input_name)[0]
    output_path = os.path.join(output_directory, stem + '.json')
    data['source'] = {'path': input_path,
                      'filename': input_name}
    logging.info(f'Writing {output_path}')
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
            logging.error(message)
    return result

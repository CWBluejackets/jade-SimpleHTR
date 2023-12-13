# library file
import logging
import os

import cv2


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


def process_sheet(filename, config):
    output_directory = config['png_directory']
    logging.info(f'Output to {output_directory}')
    logging.info(f'Reading image from {filename}')
    image = safe_imread(filename, cv2.IMREAD_GRAYSCALE, exception_if_missing=config['img_file_exception'])
    pass



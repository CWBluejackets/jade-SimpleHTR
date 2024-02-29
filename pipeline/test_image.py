#!/usr/bin/env python3
import argparse
import glob
import itertools
import json
import os

import cv2

import musters2boxes
import sheet_division


def get_filename_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Process muster sheets for ML application",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         default=musters2boxes.DEFAULT_CONFIG_FILE,
                         metavar='CONFIG.JSON',
                         help='general config file')

    oparser.add_argument('-o', dest='output_dir',
                         type=str, required=True,
                         metavar='DIR',
                         help='debug output directory')

    oparser.add_argument('-i', dest='input_location',
                         type=str, default=None,
                         metavar='FILE_OR_DIR',
                         help='input file or directory')

    options = oparser.parse_args()

    # machine-independent configuration (parameters)
    with open(options.config_file, 'r') as f:
        config = json.load(f)

    # always verbose for testing
    config['verbose'] = True

    workflow_file = os.path.join(os.path.dirname(options.config_file),
                                 config['workflows'])
    with open(workflow_file, 'r') as f:
        workflows = json.load(f)
        # JSON requires string keys; convert to int
        workflows = {int(k): v for k, v in workflows.items()}

    format_file = os.path.join(os.path.dirname(options.config_file),
                               config['formats'])
    with open(workflow_file, 'r') as f:
        formats = json.load(f)

    # Use options.input_location if specified; otherwise fall back to config file;
    # options.input_location can be a single JPG or a directory containing them

    input_location = options.input_location
    if os.path.isdir(input_location):
        extensions = ['jpg', 'JPG', 'jpeg', 'JPEG']
        file_globs = [glob.iglob(os.path.join(input_location, f'**/*.{extension}'), recursive=True)
                      for extension in extensions]
        input_files = itertools.chain.from_iterable(file_globs)
    elif os.path.isfile(input_location):
        input_files = [input_location]
    else:
        raise ValueError(f'Input location {input_location} is not a valid file or directory!')

    # Process each file and write the processed image to the debug directory
    for input_file in input_files:
        print(f'Reading {input_file}')
        v_lines, h_lines, test_image = sheet_division.process_image(input_file, config, debug=True,
                                                                    exception_if_missing=True)
        output_file = os.path.join(options.output_dir, get_filename_stem(input_file) + '_DEBUG.png')
        print(f'Writing {output_file}')
        cv2.imwrite(output_file, test_image)


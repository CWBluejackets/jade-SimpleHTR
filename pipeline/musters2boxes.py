#!/usr/bin/env python3

import argparse
import glob
import json
import logging
import os.path
from datetime import datetime

import sheet_division

oparser = argparse.ArgumentParser(description="Process muster sheets for ML application",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str, required=True,
                     metavar='config.json',
                     help='config file')

oparser.add_argument('-i', dest='input_location',
                     type=str, default=None,
                     metavar='FILE_OR_DIR',
                     help='input file or directory')

options = oparser.parse_args()

with open(options.config_file, 'r') as f:
    config = json.load(f)
    log_directory = config.get('log_directory')

    if log_directory:
        timestamp = datetime.now().isoformat(timespec='seconds')
        logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
        print(f'Logfile {logfile}')
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        print(f'Log stdout')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # TODO how to pass logging config to sheet_division.py?

workflow_file = os.path.join(os.path.dirname(options.config_file),
                             config['workflows'])
with open(workflow_file, 'r') as f:
    workflows = json.load(f)

format_file = os.path.join(os.path.dirname(options.config_file),
                           config['formats'])
with open(workflow_file, 'r') as f:
    formats = json.load(f)

# Use options.input_location if specified; otherwise fall back to config file;
# options.input_location can be a single JPG or a directory containing **/*.jpg (case-insensitive).
# TODO handle .jpeg .JPEG ?

input_location = options.input_location or config['input_directory']
if os.path.isdir(input_location):
    input_files = glob.iglob(os.path.join(input_location, '**/*.[jJ][pP][gG]'), recursive=True)
    pass
elif os.path.isfile(input_location):
    input_files = [input_location]
else:
    raise ValueError(f'Input location {input_location} is not valid!')

for input_file in input_files:
    logging.info(f'Processing {input_file}')
    sheet_division.process_sheet(input_file, config, workflows)

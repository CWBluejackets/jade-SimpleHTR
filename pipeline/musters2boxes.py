#!/usr/bin/env python3

import argparse
import glob
import itertools
import json
import logging
import os.path
from datetime import datetime

import sheet_division

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(HERE, 'config.json')


def get_metadata(input_filename):
    # TODO implement this based on accompanying information
    metadata = {'directory': os.path.dirname(input_filename),
                'filename': os.path.basename(input_filename),
                'muster_date': None}
    return metadata


oparser = argparse.ArgumentParser(description="Process muster sheets for ML application",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str, required=True,
                     default=DEFAULT_CONFIG_FILE,
                     metavar='CONFIG.JSON',
                     help='general config file')

oparser.add_argument('-l', dest='locations_file',
                     type=str, required=True,
                     metavar='LOCATIONS.JSON',
                     help='file locations config file')

oparser.add_argument('-i', dest='input_location',
                     type=str, default=None,
                     metavar='FILE_OR_DIR',
                     help='input file or directory')

options = oparser.parse_args()

# machine-dependent configuration (directories)
with open(options.location_file, 'r') as f:
    locations = json.load(f)

# machine-independent configuration (parameters)
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

    # TODO how to pass logging options to sheet_division.py?

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

# output of this program is PNGs and control files
output_dir = os.path.join(os.path.dirname(options.locations_file), locations['png_directory'])

# Use options.input_location if specified; otherwise fall back to config file;
# options.input_location can be a single JPG or a directory containing them

input_location = options.input_location or locations['input_directory']
if os.path.isdir(input_location):
    extensions = ['jpg', 'JPG', 'jpeg', 'JPEG']
    file_globs = [glob.iglob(os.path.join(input_location, f'**/*.{extension}'), recursive=True)
                  for extension in extensions]
    input_files = itertools.chain.from_iterable(file_globs)
elif os.path.isfile(input_location):
    input_files = [input_location]
else:
    raise ValueError(f'Input location {input_location} is not a valid file or directory!')

for input_file in input_files:
    logging.info(f'Processing {input_file}')
    image_metadata = get_metadata(input_file)
    sheet_division.process_sheet(input_file, image_metadata, output_dir, config, workflows, formats)


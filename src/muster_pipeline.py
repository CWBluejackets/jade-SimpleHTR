#!/usr/bin/env python3

import argparse
import glob
import itertools
import json
import os.path

from pipeline import sheet_division
from pipeline import template_handling
from pipeline.metadata_handling import Metadata
from model_nonglobal import Model

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(HERE, 'config.json')

if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Process muster sheets for ML application",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         default=DEFAULT_CONFIG_FILE,
                         metavar='CONFIG.JSON',
                         help='general config file')

    oparser.add_argument('-l', dest='locations_file',
                         type=str, required=True,
                         metavar='LOCATIONS.JSON',
                         help='file locations config file')

    oparser.add_argument('-v', dest='verbose',
                         action='store_true',
                         default=False,
                         help='verbose')

    oparser.add_argument('-i', dest='input_location',
                         type=str, default=None,
                         metavar='FILE_OR_DIR',
                         help='input file or directory')

    options = oparser.parse_args()

    # machine-dependent configuration (directory and some file locations)
    print(f'Loading {options.location_file}')
    with open(options.location_file, 'r') as f:
        locations = json.load(f)

    # machine-independent configuration (parameters)
    print(f'Loading {options.config_file}')
    with open(options.config_file, 'r') as f:
        config = json.load(f)
        config['verbose'] = options.verbose

    # workflow specifications
    workflow_file = os.path.join(os.path.dirname(options.config_file),
                                 config['workflows'])
    print(f'Loading {workflow_file}')
    with open(workflow_file, 'r') as f:
        workflows = json.load(f)
        # JSON requires string keys; convert to int
        workflows = {int(k): v for k, v in workflows.items()}

    model_control_file = os.path.join(os.path.dirname(options.locations_file),
                                      locations['model_mapping'])
    models_directory = locations['models_directory']
    print(f'Loading {model_control_file}')
    with open(model_control_file, 'r') as f:
        model_mapping = json.load(f)
    for workflow_str, subdirectory in model_mapping.items():
        workflow_id = int(workflow_str)
        model_subdirectory = os.path.join(models_directory, subdirectory)
        print(f'Loading model from {model_subdirectory}')
        model = Model(model_subdirectory)
        workflows[workflow_id]['model'] = model

    templates_file = os.path.join(os.path.dirname(options.config_file),
                                  config['templates'])
    templates_directory = os.path.join(os.path.dirname(options.config_file),
                                       config['templates_directory'])
    print(f'Loading {templates_file}')
    with open(templates_file, 'r') as f:
        templates = json.load(f)

    for key, specification in templates.items():
        template_handling.add_template_info(specification, templates_directory, workflows)

    metadata_file = os.path.join(os.path.dirname(options.location_file),
                                 locations['metadata_file'])
    print(f'Loading metadata from {metadata_file}')
    all_metadata = Metadata(metadata_file)

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

    exception_if_missing = locations['img_file_exception']

    for input_file in input_files:
        print(f'Processing {input_file}')
        image_metadata = all_metadata.get_metadata(input_file)
        sheet_division.process_muster_sheet(input_file, image_metadata, output_dir, config, workflows,
                                            templates, exception_if_missing)
        
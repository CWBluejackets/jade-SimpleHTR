#!/usr/bin/env python3

import argparse
import glob
import itertools
import json
import os.path
from collections import defaultdict, Counter

import numpy as np

import sheet_division
import template_handling
from metadata_handling import Metadata, make_index_key_from_pathname
from utils import write_sheet_info, NumpyJsonEncoder, DECILES

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

    oparser.add_argument('-d', dest='debug',
                         action='store_true',
                         default=False,
                         help='debug (save image files)')

    oparser.add_argument('-r', dest='registration_only',
                         action='store_true',
                         default=False,
                         help='registration only (no HTR)')

    oparser.add_argument('-i', dest='input_location',
                         type=str, default=None,
                         metavar='FILE_OR_DIR',
                         help='input file or directory')

    options = oparser.parse_args()

    # machine-dependent configuration (directory and file locations)
    print(f'Loading {options.locations_file}')
    with open(options.locations_file, 'r') as f:
        locations = json.load(f)

    # machine-independent configuration (HTR and geometry parameters)
    print(f'Loading {options.config_file}')
    with open(options.config_file, 'r') as f:
        config = json.load(f)
    config['verbose'] = options.verbose
    if options.registration_only:
        config['registration_only'] = True
        config['debug'] = True
    else:
        config['registration_only'] = False
        config['debug'] = options.debug
    config['htr'] = not config['registration_only']

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
        workflows[workflow_id]['model_location'] = model_subdirectory

    templates_file = os.path.join(os.path.dirname(options.config_file),
                                  config['templates'])
    templates_directory = os.path.join(os.path.dirname(options.config_file),
                                       config['template_directory'])
    print()
    print(f'Loading {templates_file}')
    with open(templates_file, 'r') as f:
        templates = json.load(f)

    for key, specification in templates.items():
        print(f'Processing template {key}')
        if config['debug']:
            debug_template_path = os.path.join(locations['output_directory'], f'template_{key}.json')
        else:
            debug_template_path = None
        template_handling.add_template_info(specification, templates_directory, workflows, debug_template_path)

    print()
    metadata_file = os.path.join(os.path.dirname(options.locations_file),
                                 locations['metadata_file'])
    inventory_file = os.path.join(os.path.dirname(options.locations_file),
                                  locations['inventory_file'])
    all_metadata = Metadata(metadata_file, inventory_file)

    # output: images and control files
    output_dir = os.path.join(os.path.dirname(options.locations_file), locations['output_directory'])
    parts_dir = os.path.join(output_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    # output: Zooniverse phase 2 for corrections where confidence score is low
    corrections_dir = os.path.join(os.path.dirname(options.locations_file), locations['corrections_directory'])
    os.makedirs(os.path.join(corrections_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(corrections_dir, 'json'), exist_ok=True)

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

    global_statistics = {'sheet_count': Counter(),
                         'sheets': defaultdict(list),
                         'cells_by_type': Counter(),
                         'cells_total': 0,
                         'corrections_by_type': Counter(),
                         'status': Counter(),
                         'status_by_sheet': defaultdict(Counter),
                         'status_by_workflow': defaultdict(Counter)}
    global_confidences = defaultdict(list)

    for input_file in input_files:
        print()
        print(f'SUMMARY: Processing {input_file}')
        index_key = make_index_key_from_pathname(input_file)
        image_metadata = all_metadata.get_metadata(index_key)
        output_prefix = os.path.splitext(os.path.basename(input_file))[0]
        sheet_info = {'input_file': input_file,
                      'index_key': index_key}
        global_statistics['sheet_count']['files found'] += 1
        if image_metadata is None:
            print(f'SUMMARY: skipping {input_file} no metadata for {index_key}')
            error = 'no metadata'
            sheet_info['error'] = error
            global_statistics['sheet_count'][error] += 1
            global_statistics['sheets'][error].append(output_prefix)
            write_sheet_info(output_dir, output_prefix, sheet_info)
        elif image_metadata['bad_alignment']:
            print(f'SUMMARY: Skipping {input_file} bad alignment')
            error = 'bad alignment'
            sheet_info['error'] = error
            global_statistics['sheet_count'][error] += 1
            global_statistics['sheets'][error].append(output_prefix)
            write_sheet_info(output_dir, output_prefix, sheet_info)
        elif image_metadata['non_standard']:
            error = 'non-standard'
            print(f'SUMMARY: Skipping {input_file}  non-standard')
            sheet_info['error'] = error
            global_statistics['sheet_count'][error] += 1
            global_statistics['sheets'][error].append(output_prefix)
            write_sheet_info(output_dir, output_prefix, sheet_info)
        else:
            sheet_division.process_muster_sheet(input_file, image_metadata,
                                                output_dir, parts_dir, corrections_dir,
                                                config, workflows,
                                                templates, global_statistics,
                                                global_confidences)
            global_statistics['sheet_count']['attempted'] += 1
            global_statistics['sheets']['attempted'].append(output_prefix)

    confidence_deciles = {str(workflow_id): np.quantile(confidence_list, DECILES)
                          for workflow_id, confidence_list
                          in global_confidences.items()}
    global_statistics['confidence_scores'] = confidence_deciles

    statistics_path = os.path.join(output_dir, 'summary.json')
    print(f'Writing {statistics_path}')
    with open(statistics_path, 'w') as f:
        json.dump(global_statistics, f, indent=1, sort_keys=True, cls=NumpyJsonEncoder)

#!/usr/bin/env python3
import csv
import json
import glob
import os.path
import argparse

MANIFEST_HEADER = ['image', 'NARA URL', 'volume', 'vessel', 'sheet', 'column',
                   '#row_no', '#column_no', '#workflow_id',
                   '#training_top', '#training_bottom', '#training_left', '#training_right',
                   '#htr_text', '#conf', '#conf_threshold',
                   'muster_date', '#muster_year'
                   ]


# VISIBLE METADATA
# image filename
# NARA URL
# volume number
# vessel name
# sheet number
# workflow name (e.g., "Eyes Hair Complexion")
# muster date
# HIDDEN METADATA
# row number
# column number
# workflow number
# coordinates of the HTR box (for extracting the image to use as new training data)
# HTR output
# muster year


def extract_year(muster_date_str, filename):
    try:
        # expecting "31/06/63"
        year = int(f'18{muster_date_str[-2:]}')
    except (ValueError, TypeError):
        year = None
        print(f'Invalid muster date in {filename}')
    return year


def process_json(json_filename1: str, image_directory1: str, csv_data1: list,
                 json_output0: dict) -> None:
    print(f'Reading {json_filename1}')
    with open(json_filename1, 'r') as f:
        json_data = json.load(f)
    image_basename = os.path.basename(json_data['correction_image'])
    image_path = os.path.join(image_directory1, image_basename)
    if not os.path.exists(image_path):
        print(f'ERROR: missing {image_path}')
    else:
        vessel = json_data['input']['vessel_raw']
        volume = json_data['input']['volume']
        sheet = json_data['input']['sheet']
        nara_url = json_data['input']['nara_url']
        workflow_id = json_data['workflow']
        workflow_name = json_data['workflow_name']
        confidence = json_data['confidence']
        confidence_threshold = json_data['confidence_threshold']
        row = json_data['row']
        column = json_data['column']
        training_top = json_data['training_boundaries']['top']
        training_bottom = json_data['training_boundaries']['bottom']
        training_left = json_data['training_boundaries']['left']
        training_right = json_data['training_boundaries']['right']
        htr_text = json_data['text']
        muster_date = json_data['input']['muster_date']
        muster_year = extract_year(muster_date, json_filename1)
        output_line = [image_basename, nara_url, volume, vessel, sheet, workflow_name,
                       row, column, workflow_id,
                       training_top, training_bottom, training_left, training_right,
                       htr_text, confidence, confidence_threshold, muster_date, muster_year]
        csv_data1.append(output_line)
        json_output0[image_basename] = dict(zip(MANIFEST_HEADER[1:], output_line[1:]))
    return


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-l', dest='locations_file',
                         type=str,
                         metavar='LOCATIONS.JSON',
                         help='file locations config file')

    oparser.add_argument('-I', dest='image_directory',
                         type=str,
                         metavar='PATH',
                         help='image and manifest directory')

    oparser.add_argument('-J', dest='json_directory',
                         type=str,
                         metavar='PATH',
                         help='json directory')

    options = oparser.parse_args()

    if not options.locations_file and not (options.json_directory and options.image_directory):
        print('You must specify either -l FILE or -I DIR and -J DIR!')
        exit(99)

    if options.locations_file:
        # machine-dependent configuration (directory and some file locations)
        print(f'Loading {options.locations_file}')
        with open(options.locations_file, 'r') as f:
            locations = json.load(f)
        corrections_directory = locations['corrections_directory']
        image_directory = os.path.join(corrections_directory, 'image')
        json_directory = os.path.join(corrections_directory, 'json')
    else:
        image_directory = options.image_directory
        json_directory = options.json_directory

    print(f'Images will be read from {image_directory}')
    print(f'JSON will be read from {json_directory}')
    json_glob = glob.iglob(os.path.join(json_directory, '*.json'))

    csv_data = []
    json_output = dict()
    for json_filename in json_glob:
        process_json(json_filename, image_directory, csv_data, json_output)

    manifest_csv_path = os.path.join(image_directory, 'manifest.csv')
    print(f'Writing {manifest_csv_path}')
    with open(manifest_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(csv_data)

    manifest_json_path = os.path.join(image_directory, 'manifest.json')
    print(f'Writing {manifest_json_path}')
    with open(manifest_json_path, 'w', newline='') as f:
        json.dump(json_output, f, indent=1, sort_keys=True)

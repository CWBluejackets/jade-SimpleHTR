#!/usr/bin/env python3
import csv
import json
import glob
import os.path
import argparse

MANIFEST_HEADER = ['image', 'NARA URL', 'volume', 'vessel', 'sheet', 'workflow',
                   '#row', '#column', '#workflow_id',
                   '#training_top', '#training_bottom', '#training_left', '#training_right',
                   '#htr_text', '#conf', '#conf_threshold'
                   ]


# VISIBLE METADATA
# image filename
# NARA URL
# volume number
# vessel name
# sheet number
# workflow name (e.g., "Eyes Hair Complexion")
# HIDDEN METADATA
# row number
# column number
# workflow number
# coordinates of the HTR box (for extracting the image to use as new training data)
# HTR output


def process_json(json_filename1: str, image_directory1: str, manifest_data1: list) -> None:
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
        output_line = [image_basename, nara_url, volume, vessel, sheet, workflow_name,
                       row, column, workflow_id,
                       training_top, training_bottom, training_left, training_right,
                       htr_text, confidence, confidence_threshold]
        manifest_data1.append(output_line)
    return


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-l', dest='locations_file',
                         type=str,
                         required=True,
                         metavar='LOCATIONS.JSON',
                         help='file locations config file')

    options = oparser.parse_args()

    # machine-dependent configuration (directory and some file locations)
    print(f'Loading {options.locations_file}')
    with open(options.locations_file, 'r') as f:
        locations = json.load(f)
    corrections_directory = locations['corrections_directory']
    image_directory = os.path.join(corrections_directory, 'image')

    json_directory = os.path.join(corrections_directory, 'json')
    json_glob = glob.iglob(os.path.join(json_directory, '*.json'))

    manifest_data = []
    for json_filename in json_glob:
        process_json(json_filename, image_directory, manifest_data)

    manifest_path = os.path.join(image_directory, 'manifest.csv')
    print(f'Writing {manifest_path}')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(manifest_data)

#!/usr/bin/env python3
import csv
import json
import glob
import os.path
import argparse

MANIFEST_HEADER = ['image', 'NARA URL', 'volume', 'vessel', 'sheet', 'type',
                   '#row', '#column', '#workflow_id',
                   '#training_top', '#training_bottom', '#training_left', '#training_right',
                   '#htr_text'
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


def process_json(json_filename1: str, image_directory: str, manifest_data1: list, workflows: dict) -> None:
    print(f'Reading {json_filename1}')
    with open(json_filename1, 'r') as f:
        json_data = json.load(f)
    image_basename = os.path.basename(json_data['output_image'])
    image_path = os.path.join(image_directory, image_basename)
    if not os.path.exists(image_path):
        print(f'ERROR: missing {image_path}')
    else:
        vessel = json_data['input']['vessel_raw']
        volume = json_data['input']['volume']
        sheet = json_data['input']['sheet']
        nara_url = json_data['input']['nara_url']
        workflow_id = json_data['workflow']
        workflow_name = workflows[str(workflow_id)]['description']
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
                       htr_text]
        manifest_data1.append(output_line)
    return


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-I', dest='image_directory',
                         type=str,
                         required=True,
                         metavar='PATH',
                         help='image input and manifest output directory')

    oparser.add_argument('-J', dest='json_directory',
                         type=str,
                         required=True,
                         metavar='PATH',
                         help='JSON input directory')

    oparser.add_argument('-w', dest='workflow_config',
                         type=str,
                         required=True,
                         metavar='WORKFLOW.JSON',
                         help='workflow JSON config file')

    options = oparser.parse_args()

    json_glob = glob.iglob(os.path.join(options.json_directory, '*.json'))
    with open(options.workflow_config, 'r') as f:
        workflow_config = json.load(f)

    manifest_data = []
    for json_filename in json_glob:
        process_json(json_filename, options.image_directory, manifest_data, workflow_config)

    manifest_path = os.path.join(options.image_directory, 'manifest.csv')
    print(f'Writing {manifest_path}')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(manifest_data)

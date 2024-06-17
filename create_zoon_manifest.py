#!/usr/bin/env python3
import csv
import json
import glob
import os.path
import argparse

MANIFEST_HEADER = ['image', 'NARA URL', 'volume', 'vessel', 'sheet', 'type',
                   '#identifier', '#row', '#column', '#workflow_id',
                   '#training_top', '#training_bottom', '#training_left', '#training_right'
                   ]

# VISIBLE METADATA
# image filename
# NARA URL
# vessel name
# sheet number
# volume number
# workflow name (e.g., "Eyes Hair Complexion")
# HIDDEN METADATA
# image identifier
# row number
# column number
# workflow number
# coordinates of the HTR box (for extracting the image to use as new training data).


def process_json(json_filename1: str, image_directory: str, manifest_data1: list) -> None:
    with open(json_filename1, 'r') as f:
        json_data = json.load(f)
    vessel = json_data['vessel_raw']
    volume = json_data['volume']
    sheet = json_data['sheet']
    nara_url = json_data['nara_url']





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
        writer = csv.DictWriter(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(manifest_data)

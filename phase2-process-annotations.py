#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
import json
import logging
import os
from collections import defaultdict

import pandas as pd


def process_zoon_row(csv_row):
    output = {}
    for k, v in csv_row.items():
        if k in ['metadata', 'subject_data']:
            output[k] = json.loads(v)
        elif k == 'annotations':
            output[k] = process_zoon_annotations(v)
        else:
            output[k] = v
    finish_time = output['metadata']['finished_at'].split('.')[0]
    output['timestamp'] = datetime.strptime(finish_time, '%Y-%m-%dT%H:%M:%S')
    # '2024-01-17T12:53:54.831Z'
    return output


def process_zoon_annotations(annotation_list_json):
    annotation_list = json.loads(annotation_list_json)
    output = defaultdict(None)
    for annotation1 in annotation_list:
        task = annotation1['task']
        if task in ['T0', 'T1']:
            output[annotation1['task']] = annotation1['value']
    return output


oparser = argparse.ArgumentParser(description="analyse the annotations processed from the exports",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-i', dest='input_file',
                     type=str,
                     required=True,
                     metavar='INPUT.csv',
                     help='input file')

oparser.add_argument('-o', dest='output_file',
                     type=str,
                     required=True,
                     metavar='OUTPUT.json',
                     help='output file')

options = oparser.parse_args()

input_data = []

with open(options.input_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        input_data.append(process_zoon_row(row))

data_by_subject = defaultdict(list)
for item in input_data:
    data_by_subject[item['subject_id']].append(item)

#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
import json
import logging
import os
from collections import defaultdict, Counter

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
    output = {}
    for annotation1 in annotation_list:
        task = annotation1['task']
        if task in ['T0', 'T1']:
            output[annotation1['task']] = annotation1['value']
    return output


def reconcile_annotations(classifications1):
    task_values = defaultdict(list)
    for classification1 in classifications1:
        for k, v in classification1['annotations'].items():
            task_values[k].append(v)
    if vote(task_values['T0']) == 'Yes':
        result = True, ''
    else:
        result = False, vote(task_values['T1'])
    return result


def vote(things):
    if len(things) == 0:
        result = None
    elif len(things) == 1:
        result = things[0]
    else:
        c = Counter(things)
        result = c.most_common(1)[0][0]
    # crude, doesn't handle ties
    return result


def extra_filename(fn):
    dir = os.path.dirname(fn)
    base, suffix = os.path.splitext(os.path.basename(fn))
    return os.path.join(dir, f'{base}_debug.{suffix}')


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

debug_file = extra_filename(options.output_file)

input_data = []

with open(options.input_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        input_data.append(process_zoon_row(row))

data_by_subject = defaultdict(list)
for item in input_data:
    data_by_subject[item['subject_id']].append(item)

full_output = {}
corrections = {}

for subject, classifications in data_by_subject.items():
    was_correct, correction = reconcile_annotations(classifications)
    full_output[subject] = [was_correct, correction, classifications]
    if not was_correct:
        corrections[subject] = correction

with open(options.output_file, 'w') as f:
    json.dump(corrections, f, indent=1)

with open(debug_file, 'w') as f:
    json.dump(full_output, f, indent=1)

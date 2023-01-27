#!/usr/bin/env python3
import argparse
import csv
import datetime
import glob
import json
import logging
import math
import os
import re

import pandas as pd

DEFAULT_OUTPUT_FILENAME = os.path.join(os.environ['HOME'], 'output.csv')
TASK_RE = re.compile(r'T(\d+)')

# output format CSV
COLUMNS = ['subject_id', 'classification_id', 'workflow_id', 'annotator', 'filename',
           'finished_str', 'task', 'x', 'y', 'width', 'height', 'text', 'text_clean',
           'left', 'right', 'top', 'bottom']


WHITESPACE = re.compile(r'\s+')


def add_coordinates(row0):
    '''
    add left, right, top, bottom values to a row
    :param row0: row of annotation data
    :return: original row extended with four more values
    '''
    x = row0[COLUMNS.index('x')]
    y = row0[COLUMNS.index('y')]
    width = row0[COLUMNS.index('width')]
    height = row0[COLUMNS.index('height')]
    if x and width:
        left = math.floor(x)
        right = math.ceil(x + width)
    else:
        left, right = None, None
    if y and height:
        top = math.floor(y)
        bottom = math.ceil(y + height)
    else:
        top, bottom = None, None
    return row0 + [left, right, top, bottom]


def clean_string(s0: str):
    if not s0:
        return ' '
    # Empty strings need to be changed to one space to prevent automatic
    # conversion to NaN through CSV interchange and to allow HTR to work.
    s1 = WHITESPACE.sub(' ', s0.strip())
    s1 = s1 or ' '
    # Some input contains literally "N/A", which pandas turns into Nan
    # ignoring this for now
    return s1


def convert_t_code(task_id):
    """
    Validate task ID and return numeric value
    :param task_id: "T11"
    :return: 11 (int)
    """
    match = TASK_RE.match(task_id)
    if not match:
        raise ValueError(f'invalid task id {task_id}')
    return int(match.group(1))


def process_row(row0: pd.Series):
    results0 = []
    classification_id = row0['classification_id']
    workflow_id = row0['workflow_id']
    annotator = row0['user_name']
    subject_id = row0['subject_ids']
    subject_data = json.loads(row0['subject_data'])
    filename = subject_data[str(subject_id)].get('Filename')
    metadata = json.loads(row0['metadata'])
    finished_str = metadata.get('finished_at')
    # annotations = list of things
    annotations = json.loads(row0['annotations'])
    for annotation in annotations:
        partial_results = process_annotation(annotation)
        for partial in partial_results:
            # partial = task, x, y, width, height, text, text_clean,
            full_result = [subject_id, classification_id, workflow_id, annotator,
                           filename, finished_str] + partial
            results0.append(full_result)
    return results0


def process_annotation(annotation):
    """
    :param annotation:
    :return: list of lists without the filenames
    """
    task_str = annotation.get('task', 'MISSING0')
    value = annotation.get('value')
    # task, x, y, width, height, text, text_clean
    if value is None:
        results1 = [[task_str, None, None, None, None, None, ' ']]
    elif isinstance(value, str):
        results1 = [[task_str, None, None, None, None, value, clean_string(value)]]
    # otherwise, it should be a list
    else:
        results1 = []
        for item in value:
            results1.extend(process_sub_value(item))
    return results1


def process_sub_value(item):
    results1 = []
    task_str = item.get('task', 'MISSING1')
    sublist = item.get('value', [])
    for item1 in sublist:
        x = item1.get('x')
        y = item1.get('y')
        width = item1.get('width')
        height = item1.get('height')
        for detail in item1.get('details', []):
            value = detail.get('value')
            result1 = [task_str, x, y, width, height, value, clean_string(value)]
            results1.append(result1)
    return results1


oparser = argparse.ArgumentParser(description="Convert classifications CSV",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     required=True, metavar='FILE.json',
                     help='config file')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

timestamp = datetime.datetime.now().isoformat(timespec='seconds')
logfile = os.path.join(config['log_directory'], f'{timestamp}_convert.log')
print('Logfile', logfile)
logging.basicConfig(filename=logfile, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

results = []

input_pattern = os.path.join(config['input_directory'], '**', '**.csv')
logging.warning(f'Seeking input {input_pattern}')
for input_filename in glob.iglob(input_pattern, recursive=True):
    logging.info(f'Reading {input_filename}')
    input_df = pd.read_csv(input_filename)
    for index, row in input_df.iterrows():
        row_results = process_row(row)
        results.extend(row_results)

# add left, right, top, bottom
results = [add_coordinates(result) for result in results]

output_file = config['intermediate_file']
with open(output_file, 'w', newline='') as f:
    logging.warning(f'Writing {output_file}')
    fw = csv.writer(f)
    fw.writerow(COLUMNS)
    for result in results:
        fw.writerow(result)

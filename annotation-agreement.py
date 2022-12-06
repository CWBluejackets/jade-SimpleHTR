#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
from collections import defaultdict

import pandas as pd


def process_big_df(dataframe):
    # delete the rows without rectangles, text, or identifiers
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text; filtered rows {dataframe.shape[0]}')
    # put all the annotations into a quasi-database
    # key: tuple(subject, workflow, line)
    # value: dict annotator -> annotation
    annotation_store0 = defaultdict(dict)

    # split into subunits by subject and workflow IDs and annotator
    grouped_columns = dataframe.groupby(by=['subject_id', 'workflow_id', 'annotator'])
    for muster_column in grouped_columns.groups.keys():
        logging.info(f'Processing {muster_column}')
        muster_column_raw = grouped_columns.get_group(muster_column)
        # create a new DF so we can reset the index and add a column
        muster_column_df = muster_column_raw.sort_values(by='y', ascending=True)
        # index must be reset for the pd range sequence to work
        muster_column_df.reset_index()
        muster_column_df['line'] = pd.Series(range(muster_column_df.shape[0]))
        for i, row in muster_column_df.iterrows():
            key = (row['subject_id'], row['workflow_id'], row['line'])
            annotator = row['annotator']
            annotation = row['text']
            annotation_store0[key][annotator] = annotation
    logging.warning(f'Number of muster lines f{len(annotation_store0)}')
    return annotation_store0


oparser = argparse.ArgumentParser(description="IAA and stuff",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str,
                     required=True,
                     metavar='config.json',
                     help='config file')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

# turn off the chained assignment warnings
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

log_directory = config.get('log_directory')

if log_directory:
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
    print(f'Logfile {logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:
    print(f'Log stdout')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

input_filename = config['intermediate_file']
output_filename = config['agreed_file']

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)

annotation_store = process_big_df(big_df)
# TODO process that


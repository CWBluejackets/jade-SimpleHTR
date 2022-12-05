#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os

import pandas as pd


def process_big_df(dataframe):
    # delete the rows without rectangles, text, or identifiers
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text; filtered rows {dataframe.shape[0]}')

    # split into subunits by subject and workflow IDs
    grouped_columns = dataframe.groupby(by=['subject_id', 'workflow_id'])
    for muster_column in grouped_columns.groups.keys():
        muster_column_df = grouped_columns.get_group(muster_column)
        # split by annotator
        grouped_annotators = muster_column_df.groupby(by='annotator')
        annotators = muster_column_df['annotator'].unique()
        for annotator in annotators:
            column_annotator_df = grouped_annotators.get_group(annotator)
            # sort vertically on the page
            column_annotator_df.sort_values(by='y', inplace=True, ascending=True)

    # ...
    return


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

process_big_df(big_df)

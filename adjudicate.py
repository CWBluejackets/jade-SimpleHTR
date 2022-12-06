#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
from collections import defaultdict, Counter

import pandas as pd

import create_training_data


def process_big_df(dataframe):
    # delete the rows without rectangles, text, or identifiers
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text; filtered rows {dataframe.shape[0]}')

    # clean the ends of the strings and suppress newlines
    dataframe['text'] = dataframe['text'].apply(create_training_data.clean_string)
    logging.warning(f'Cleaned text strings')

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


def process_database(annotation_store0: dict):
    # key: tuple(subject, workflow, line)
    # annotation_store value: dict annotator -> annotation
    # decisions value: annotation or None if no majority
    decisions0 = dict()
    for key, annotation_dict in annotation_store0.items():
        result = decide(key, annotation_dict)
        decisions0[key] = result
        # store None to be ignored later
    return decisions0


def decide(key_triple: tuple, annotation_dict: dict):
    c = Counter(annotation_dict.values())
    mode, votes = c.most_common(1)[0]
    if votes > len(c) / 2:
        # majority :-)
        return mode
    # implied else: log as dodgy
    logging.info(f'Voting failed: {key_triple} has {c.most_common()}')
    return None


def apply_decisions(dataframe, decisions0):
    dataframe['decision'] = dataframe.apply(apply_one_decision, axis=1, decision_dict=decisions0)
    return dataframe


def apply_one_decision(row: pd.Series, decision_dict: dict):
    key = (row['subject_id'], row['workflow_id'], row['line'])
    return decision_dict.get(key)


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
    logfile = os.path.join(config['log_directory'], f'{timestamp}_adjudicate.log')
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
decisions = process_database(annotation_store)

big_df1 = apply_decisions(big_df, decisions)
logging.warning(f'Writing {output_filename}')
big_df1.to_csv(output_filename)


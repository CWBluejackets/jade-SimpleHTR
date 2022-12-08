#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
from collections import defaultdict, Counter

import pandas as pd

import create_training_data


def process_big_df(dataframe, options0):
    logging.warning(f'Starting rows {dataframe.shape[0]}')
    # delete the rows without rectangles, text, or identifiers
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text/IDs; filtered rows {dataframe.shape[0]}')

    # clean the ends of the strings and suppress newlines
    dataframe['text'] = dataframe['text'].apply(create_training_data.clean_string)
    logging.warning(f'Cleaned text strings')
    # create column with special value to start it as int64
    dataframe['line'] = -1

    # put all the annotations into a quasi-database
    # key: tuple(subject, workflow, line)
    # value: dict annotator -> annotation
    annotation_store0 = defaultdict(dict)

    # split into subunits by subject and workflow IDs and annotator
    grouped_columns = dataframe.groupby(by=['subject_id', 'workflow_id', 'annotator'])
    for muster_column in grouped_columns.groups.keys():
        muster_column_raw = grouped_columns.get_group(muster_column)
        logging.info(f'Processing {muster_column} with {muster_column_raw.shape[0]} rows')
        # create a new DF so we can reset the index and add a column
        muster_column_df = muster_column_raw.sort_values(by='y', ascending=True)
        # index must be reset for the pd range sequence to work
        muster_column_df.reset_index(inplace=True, drop=True)
        muster_column_df['line'] = pd.Series(range(muster_column_df.shape[0]))
        for i, row in muster_column_df.iterrows():
            subject_id = row['subject_id']
            workflow_id = row['workflow_id']
            line = row['line']
            y = row['y']
            # add this manual annotation to the annotation store
            key = (subject_id, workflow_id, line)
            annotator = row['annotator']
            annotation = row['text']
            annotation_store0[key][annotator] = annotation
            # add the line number back to the main dataframe for use later
            main_index = dataframe.index[(dataframe['subject_id'] == subject_id) &
                                         (dataframe['workflow_id'] == workflow_id) &
                                         (dataframe['annotator'] == annotator) &
                                         (dataframe['y'] == y)
                                         ].tolist()[0]
            dataframe.at[main_index, 'line'] = line
    logging.warning(f'Total number of annotated muster lines {len(annotation_store0)}')
    return annotation_store0


def process_database(annotation_store0: dict):
    # key: tuple(subject, workflow, line)
    # annotation_store value: dict annotator -> annotation
    # decisions value: tuple(annotation or None, mode votes, total votes)
    decisions0 = dict()
    # decision type, number of votes -> nbr of matching decisions
    decision_counter = defaultdict(Counter)
    for key, annotation_dict in annotation_store0.items():
        voting_results = decide(key, annotation_dict, decision_counter)
        decisions0[key] = voting_results
        # decision = None means no majority so no result
    total = 0
    for vote_type, stuff in decision_counter.items():
        subtotal = 0
        for vote_size, nbr in stuff.most_common():
            subtotal += nbr
            logging.warning(f'Decision type {vote_type} from {vote_size} = {nbr}')
        logging.warning(f'Decision type {vote_type} subtotal = {subtotal}')
        total += subtotal
    logging.warning(f'Decision total = {total}')
    return decisions0


def decide(key_triple: tuple, annotation_dict: dict, decision_counter: dict):
    c = Counter(annotation_dict.values())
    mc = c.most_common()
    mode, mode_votes = mc[0]
    total_votes = len(annotation_dict)
    if mode_votes == total_votes:
        decision_counter['unanimous'][total_votes] += 1
        result = mode
    elif mode_votes > total_votes / 2:
        decision_counter['majority'][total_votes] += 1
        logging.info(f'Voting majority: {key_triple} has {c.most_common()}')
        logging.info(f'  from {annotation_dict}')
        result = mode
    elif mode_votes > c.most_common()[1][1]:
        decision_counter['plurality'][total_votes] += 1
        logging.info(f'Voting plurality: {key_triple} has {c.most_common()}')
        logging.info(f'  from {annotation_dict}')
        result = mode
    else:  # dodgy
        decision_counter['failed'][total_votes] += 1
        logging.info(f'Voting failed: {key_triple} has {c.most_common()}')
        logging.info(f'  from {annotation_dict}')
        result = None
    return result, mode_votes, total_votes


def apply_decisions(dataframe, decisions0):
    dataframe = dataframe.join(dataframe.apply(apply_one_decision, axis=1, decision_dict=decisions0))
    logging.warning(f"filled in decision = {dataframe['decision'].count()}")
    logging.warning(f"filled in line = {dataframe['line'].count()}")
    logging.warning(f'Dataframe shape = {dataframe.shape}')
    return dataframe


def apply_one_decision(row: pd.Series, decision_dict: dict):
    key = (row['subject_id'], row['workflow_id'], row['line'])
    result, mode_votes, total_votes = decision_dict.get(key, (None, 0, 0))
    # TODO force mode_votes and total_votes to int
    return pd.Series([result, mode_votes, total_votes], index=['decision', 'mode_votes', 'total_votes'])


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="IAA and stuff",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         required=True,
                         metavar='config.json',
                         help='config file')

    oparser.add_argument('-d', dest='debug',
                         default=False,
                         action='store_true',
                         help='debug')

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

    annotation_store = process_big_df(big_df, options)
    decisions = process_database(annotation_store)

    big_df1 = apply_decisions(big_df, decisions)
    logging.warning(f'Writing {output_filename}')
    big_df1.to_csv(output_filename)

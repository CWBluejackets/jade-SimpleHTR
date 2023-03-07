#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os

import pandas as pd
from pandas import DataFrame


def process_big_df(dataframe: DataFrame, output_directory0, filters0: dict):
    """
    Process the dataframe from our intermediate CSV format to generate training data
    :param filters0: map of workflow -> regex
    :param output_directory0:
    :param dataframe:
    :return:
    """
    logging.warning(f'Unfiltered rows {dataframe.shape[0]}')

    # Use the adjudicated data if present.
    if 'decision1' in dataframe.columns:
        text_column = 'decision1'
    elif 'decision' in dataframe.columns:
        text_column = 'decision'
    else:
        text_column = 'text_clean'

    # delete the rows without rectangles or text
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', text_column])
    logging.warning(f'Dropped NaN rectangles/text; rows left {dataframe.shape[0]}')

    # if the 'ignored' column is present, delete rows that were ignored in voting
    if 'wrong_column' in dataframe.columns:
        dataframe = dataframe[~ dataframe['wrong_column']]
        logging.warning(f'Dropped wrong column annotations; rows left {dataframe.shape[0]}')
    else:
        logging.warning(f'Column "wrong_column" not present')

    if 'too_small' in dataframe.columns:
        dataframe = dataframe[~ dataframe['too_small']]
        logging.warning(f'Dropped boxes that are too small; rows left {dataframe.shape[0]}')
    else:
        logging.warning(f'Column "too_small" not present')

    # split into separate dataframes by workflow_id
    grouped = dataframe.groupby(dataframe['workflow_id'])

    for wf_id in dataframe['workflow_id'].unique():
        sub_dataframe = grouped.get_group(wf_id)
        logging.info(f'Workflow_id {wf_id} has {sub_dataframe.shape[0]} rows')

        # optional: apply regex filter to text; let everything through if no filter is specified
        wf_pattern = filters0.get(wf_id)
        if wf_pattern:
            # save a DF of regex rejects in the output directory (with workflow ID in the filename)
            rejected_df = sub_dataframe[~ sub_dataframe[text_column].str.fullmatch(wf_pattern)]
            rejected_path = os.path.join(output_directory0, f'filtered_{wf_id}.csv')
            rejected_df.to_csv(rejected_path)
            # and a count of the rejected strings
            reject_counter = rejected_df[text_column].value_counts()
            reject_count_path = os.path.join(output_directory0, f'filtered_count_{wf_id}.csv')
            reject_counter.to_csv(reject_count_path)
        else:
            logging.info(f'No filter for this workflow')
    return


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate training data",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         required=True,
                         metavar='config.json',
                         help='config file')

    oparser.add_argument('-o', dest='output_directory',
                         type=str,
                         default='.',
                         metavar='DIR',
                         help='output directory')

    oparser.add_argument('-f', dest='filter_files',
                         default=[],
                         action='append',
                         metavar='FILTER.json',
                         help='filter file (can be repeated)')

    options = oparser.parse_args()

    with open(options.config_file) as f:
        config = json.load(f)

    # turn off the chained assignment warnings
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    pd.options.mode.chained_assignment = None

    log_directory = config.get('log_directory')

    if log_directory:
        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        logfile = os.path.join(config['log_directory'], f'{timestamp}_filter.log')
        print(f'Logfile {logfile}')
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        print(f'Log stdout')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    agreed_filename = config['agreed_file']
    output_directory = options.output_directory
    os.makedirs(output_directory, exist_ok=True)

    if os.path.exists(agreed_filename):
        input_filename = agreed_filename
    else:
        input_filename = config['intermediate_file']

    filters = {}

    for filter_file in options.filter_files:
        with open(filter_file, 'r') as f:
            addenda = json.load(f)
            # force keys to be int for compatibility later
            filters.update({int(k): v for k, v in addenda.items()})

    big_df = pd.read_csv(input_filename)
    process_big_df(big_df, output_directory, filters)

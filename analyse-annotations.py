#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import logging
import math
import os
import re
import uuid
from collections import Counter
from typing import Tuple, Dict

import pandas as pd
from PIL import Image
from pandas import Series, DataFrame

NEWLINES = re.compile('\n+')


def clean_string(s0: str):
    s1 = s0.strip()
    return NEWLINES.sub(' ', s1)


def process_big_df(dataframe: DataFrame, output_directory0, jpg_index0: dict,
                   stem0: str, max_size0: int, samples0: list, filters0: dict):
    """
    Process a dataframe from our CSV format to generate training data
    :param filters0: map of workflow -> regex
    :param samples0:
    :param max_size0:
    :param output_directory0:
    :param dataframe:
    :param jpg_index0: the JPG file index
    :param stem0: starter for the output filenames
    :return:
    """
    logging.warning(f'Unfiltered rows {dataframe.shape[0]}')

    # delete the rows without rectangles or text
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text'])
    logging.warning(f'Dropped NaN rectangles/text; filtered rows {dataframe.shape[0]}')

    # delete rows with jpg filenames that are not available
    dataframe = dataframe[dataframe['filename'].isin(jpg_index0.keys())]
    logging.warning(f'Dropped unavailable jpgs; filtered rows {dataframe.shape[0]}')

    # clean the ends of the strings and suppress newlines
    dataframe['text'] = dataframe['text'].apply(clean_string)
    logging.warning(f'Cleaned text strings')

    # split into separate dataframes by workflow_id
    grouped = dataframe.groupby(dataframe['workflow_id'])
    workflow_index = make_workflow_index(dataframe, samples0, output_directory0)

    for wf_id in dataframe['workflow_id'].unique():
        sub_dataframe = grouped.get_group(wf_id)
        logging.info(f'Workflow_id {wf_id} has {sub_dataframe.shape[0]} rows')

        # optional: apply regex filter to text; let everything through if no filter is specified
        wf_pattern = filters0.get(wf_id)
        if wf_pattern:
            sub_dataframe = sub_dataframe[sub_dataframe['text'].str.fullmatch(wf_pattern)]
            logging.info(f'Filter {wf_pattern} passes {sub_dataframe.shape[0]} rows')

        # optional: constrain maximum dataset size:
        if max_size0:
            if max_size0 < sub_dataframe.shape[0]:
                sub_dataframe = sub_dataframe.sample(n=max_size0)
            logging.warning(f'Max size {max_size0} applied; filtered rows {sub_dataframe.shape[0]}')

        for sample in samples0:
            wf_df = sub_dataframe.sample(frac=sample).reset_index()
            workflow_directory = workflow_index[(wf_id, sample)]
            logging.info(f'Starting workflow {wf_id} with {wf_df.shape[0]} rows (sampled at {sample})')
            process_workflow_df(wf_df, workflow_directory, jpg_index0, stem0)
    return


def process_workflow_df(dataframe: DataFrame, output_subdirectory: str, jpg_index0: dict, stem0: str):
    os.makedirs(output_subdirectory, exist_ok=True)
    info_subdirectory = os.path.join(output_subdirectory, 'info')
    os.makedirs(info_subdirectory, exist_ok=True)
    img_subdirectory = os.path.join(output_subdirectory, 'img')
    os.makedirs(img_subdirectory, exist_ok=True)

    words_file_data = []
    word_counter = Counter()
    words_file_path = os.path.join(output_subdirectory, 'words.txt')
    count_file_path = os.path.join(output_subdirectory, 'word_count.tsv')
    filename_prefix = uuid.uuid4().hex

    # group by filename so we load each JPG file only once
    grouped_df = dataframe.groupby(dataframe['filename'])

    for filename in dataframe['filename'].unique():
        small_image, big_image = load_images(filename, jpg_index0)
        small_df = grouped_df.get_group(filename)

        for index, row in small_df.iterrows():
            cropped_image = crop_image(small_image, big_image, row)
            info_path = os.path.join(output_subdirectory, 'info', f'{filename_prefix}_{index:07d}.csv')
            png_filename = f'{filename_prefix}_{index:07d}.png'
            png_path = os.path.join(img_subdirectory, png_filename)
            words_file_data.append(generate_words_row(png_filename, row))
            word_counter[row['text']] += 1
            save_shtr_files(cropped_image, row, info_path, png_path)

    # write the words.txt file and the annotation count
    with open(words_file_path, 'w') as f0:
        words_file_data.sort()
        for line in words_file_data:
            f0.write(line)
    logging.info(f'Saved words file: {words_file_path}')
    logging.info(f'Number of entries: {len(words_file_data)}')
    count_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['count'])
    count_df.sort_values(inplace=True, by='count', ascending=False)
    count_df.to_csv(count_file_path, sep='\t')
    logging.info(f'Saved count file: {count_file_path}')
    return


oparser = argparse.ArgumentParser(description="analyse the annotations processed from the exports",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str,
                     required=True,
                     metavar='FILE.json',
                     help='config file')

oparser.add_argument('-o', dest='output_file',
                     type=str,
                     required=True,
                     metavar='FILE.csv',
                     help='output file')

oparser.add_argument('-i', dest='interval',
                     default=None,
                     metavar='INTERVAL',
                     help='optional column interval')

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
    print(f'Logfile f{logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:
    print(f'Log stdout')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

input_filename = config['intermediate_file']

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)

#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import logging
import math
import os
import random
import uuid
from collections import Counter
from typing import Tuple, Dict

import pandas as pd
from PIL import Image
from pandas import Series, DataFrame


def get_jpg_filenames(small_jpg_directory: str, big_jpg_directory: str) -> Dict[str, Tuple[str, str]]:
    """
    Find and index all the *.jpg files in the specified directories.
    :param big_jpg_directory: contains the original scanned JPGs
    :param small_jpg_directory: contains the JPGs reduced below 1 MB for Zooniverse
    :return: index from basename to tuple[2] of full paths with None in the second position if
    the big JPG is not available
    """
    small_pattern = os.path.join(small_jpg_directory, '**', '*.jpg')
    logging.info(f'Using {small_pattern}')
    jpg_glob = glob.iglob(small_pattern, recursive=True)
    small_index = {os.path.basename(x): x for x in jpg_glob}
    logging.info(f'Found {len(small_index)} jpg files')

    if big_jpg_directory:
        big_pattern = os.path.join(big_jpg_directory, '**', '*.jpg')
        logging.info(f'Using {big_pattern}')
        big_glob = glob.iglob(big_pattern, recursive=True)
        big_index = {os.path.basename(x): x for x in big_glob}
        logging.info(f'Found {len(big_index)} jpg files')
    else:
        big_index = dict()

    index = {key: (small_path, big_index.get(key)) for key, small_path in small_index.items()}
    return index


def identify_big_jpgs(jpg_index0: dict) -> set:
    """
    find the available big JPG filenames
    :param jpg_index0: index from get_jpg_filenames(...)
    :return: set of available big JPG keys
    """
    return {key for key in jpg_index0.keys() if jpg_index0[key][1]}


def make_workflow_index(dataframe: DataFrame, main_output_directory: str) -> dict:
    """
    Index and create output subdirectories for the workflows and sampling rates
    :param dataframe: dataframe with a 'workflow' column
    :param main_output_directory:
    :return: index from workflow ID and sampling to full path
    """
    workflows = dataframe['workflow_id'].unique()
    index = dict()
    for workflow in workflows:
        # subdirectory, e.g., "23456_0.75" for workflow and sampling rate
        wf_directory = os.path.join(main_output_directory, str(workflow))
        index[workflow] = wf_directory
        logging.info(f'Workflow directory {wf_directory}')
        os.makedirs(wf_directory, exist_ok=True)
    return index


def process_big_df(dataframe: DataFrame, output_directory0, jpg_index0: dict,
                   max_size0: int, samples0: list, filters0: dict, holdout0: float):
    """
    Process the dataframe from our intermediate CSV format to generate training data
    :param holdout0:
    :param filters0: map of workflow -> regex
    :param samples0:
    :param max_size0:
    :param output_directory0:
    :param dataframe:
    :param jpg_index0: the JPG file index
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

    # identify missing jpg files (both sizes)
    jpgs_requested = set(dataframe['filename'].unique())
    missing_small_jpgs = sorted(jpgs_requested - jpg_index0.keys())
    missing_big_jpgs = sorted(jpgs_requested - identify_big_jpgs(jpg_index0))

    missing_small_path = os.path.join(output_directory0, 'missing_small_jpgs.txt')
    logging.info(f'Missing small jpgs {len(missing_small_jpgs)} -> {missing_small_path}')
    with open(missing_small_path, 'w') as f:
        for k in missing_small_jpgs:
            f.write(f'{k}\n')

    missing_big_path = os.path.join(output_directory0, 'missing_big_jpgs.txt')
    logging.info(f'Missing big jpgs {len(missing_big_jpgs)} -> {missing_big_path}')
    with open(missing_big_path, 'w') as f:
        for k in missing_big_jpgs:
            f.write(f'{k}\n')

    # delete rows with jpg filenames that are not available
    dataframe = dataframe[dataframe['filename'].isin(jpg_index0.keys())]
    logging.warning(f'Dropped unavailable jpgs; filtered rows {dataframe.shape[0]}')

    # split into separate dataframes by workflow_id
    grouped = dataframe.groupby(dataframe['workflow_id'])
    workflow_index = make_workflow_index(dataframe, output_directory0)

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
            # now keep the ones that pass the filters
            sub_dataframe = sub_dataframe[sub_dataframe[text_column].str.fullmatch(wf_pattern)]
            logging.info(f'Filter {wf_pattern} passes {sub_dataframe.shape[0]} rows')
        else:
            logging.info(f'No filter for this workflow')

        # optional: constrain maximum dataset size:
        if max_size0:
            if max_size0 < sub_dataframe.shape[0]:
                sub_dataframe = sub_dataframe.sample(n=max_size0)
            logging.warning(f'Max size {max_size0} applied; filtered rows {sub_dataframe.shape[0]}')

        workflow_directory = workflow_index[wf_id]
        process_workflow_df(sub_dataframe, workflow_directory, jpg_index0, text_column, samples0, holdout0)
    return


def process_workflow_df(dataframe: DataFrame, output_subdirectory: str, jpg_index0: dict, text_column: str,
                        samples0: list, holdout0: float):
    """
    Process the dataframe for one workflow ID, already sampled
    :param text_column: use 'decision' or 'text_clean
    :param dataframe: dataframe for selected ID
    :param output_subdirectory: e.g., path/to23038_0.75
    :param jpg_index0: index of JPGs
    :return:
    """
    info_subdirectory = os.path.join(output_subdirectory, 'info')
    os.makedirs(info_subdirectory, exist_ok=True)
    img_subdirectory = os.path.join(output_subdirectory, 'img')
    os.makedirs(img_subdirectory, exist_ok=True)

    words_file_data = []
    word_counter = Counter()
    count_file_path = os.path.join(output_subdirectory, 'annotation_count.tsv')
    filename_prefix = uuid.uuid4().hex

    # Group by filename so we load each JPG file only once
    grouped_df = dataframe.groupby(dataframe['filename'])

    for filename in dataframe['filename'].unique():
        small_image, big_image = load_images(filename, jpg_index0)
        small_df = grouped_df.get_group(filename)

        for index, row in small_df.iterrows():
            cropped_image = crop_image(small_image, big_image, row)
            info_path = os.path.join(output_subdirectory, 'info', f'{filename_prefix}_{index:07d}.csv')
            png_filename = f'{filename_prefix}_{index:07d}.png'
            png_path = os.path.join(img_subdirectory, png_filename)
            words_file_data.append(generate_words_row(png_filename, row, text_column))
            word_counter[row[text_column]] += 1
            save_shtr_files(cropped_image, row, info_path, png_path)

    # carry out holdout and sampling on the control files so we write the images only once
    # optional: holdout
    if holdout0:
        random.shuffle(words_file_data)
        holdout_size = math.ceil(len(words_file_data) * holdout0)
        holdout_data = words_file_data[:holdout_size]
        words_file_data = words_file_data[holdout_size:]
        save_words_file(output_subdirectory, 'holdout', holdout_data)

    for sample in samples0:
        sample_count = math.ceil(len(words_file_data) * sample)
        sample_suffix = f'{sample:.2f}'
        sample_data = random.sample(words_file_data, sample_count)
        save_words_file(output_subdirectory, sample_suffix, sample_data)

    count_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['count'])
    count_df.sort_values(inplace=True, by='count', ascending=False)
    count_df.to_csv(count_file_path, sep='\t')
    logging.info(f'Saved count file: {count_file_path}')
    return


def save_words_file(directory, suffix: str, data: list):
    filename = os.path.join(directory, f'words_{suffix}.txt')
    with open(filename, 'w') as f0:
        for line in sorted(data):
            f0.write(line)
    logging.info(f'Saved words file: {filename} with {len(data)} entries')
    return


def generate_words_row(png_filename, row: Series, text_column: str):
    """
    Produce one line for the simplified words.txt format
    :param text_column: use 'decision' or 'text_clean
    :param png_filename:
    :param row:
    :return:
    """
    # refer to SimpleHTR.md for the meaning of these
    text = row[text_column]
    line = f'{png_filename} {text}\n'
    return line


def crop_image(small_image: Image, big_image: Image, row: Series) -> Image:
    """
    Crop the image according to the coordinates in the annotation.
    :param big_image:
    :param small_image:
    :param row:
    :return: cropped image
    """
    # Both PIL and Zooniverse: (0, 0) is top left; all coordinates in pixels.
    # Zooniverse box: x, y = top left corner of box; and use width and height.
    # PIL box: tuple (left, top, right, bottom) absolute locations within image.
    # To be careful, round x and y down; width and height up.
    if big_image:
        left = math.floor(row['x'] * big_image.width / small_image.width)
        top = math.floor(row['y'] * big_image.height / small_image.height)
        right = math.ceil((row['x'] + row['width']) * big_image.width / small_image.width)
        bottom = math.ceil((row['y'] + row['height']) * big_image.height / small_image.height)
        cropped = big_image.crop(box=(left, top, right, bottom))
    else:
        left = math.floor(row['x'])
        top = math.floor(row['y'])
        right = math.ceil(left + row['width'])
        bottom = math.ceil(top + row['height'])
        cropped = small_image.crop(box=(left, top, right, bottom))
    return cropped


def save_shtr_files(cropped_image: Image, row: Series, info_file_path: str, png_path: str):
    """
    Save all the files
    :param png_path:
    :param info_file_path:
    :param cropped_image:
    :param row:
    :return:
    """
    row['png_path'] = png_path
    row.to_csv(info_file_path)
    logging.info(f'Saved info {info_file_path}')
    ''

    cropped_image.save(png_path, bits=8)
    logging.info(f'Saved image {png_path}')
    return


def load_images(basename: str, jpg_index0) -> Tuple[Image.Image, Image.Image]:
    """
    Locate a jpg file from its basename and load it
    :param basename: 'foo.jpg'
    :param jpg_index0: the index
    :return: PIL image
    """
    small_path, big_path = jpg_index0[basename]
    small_image = Image.open(small_path, 'r')
    if big_path:
        big_image = Image.open(big_path, 'r')
        # check that the aspect ratios are close
        expected_width = big_image.width * small_image.height / big_image.height
        if abs(expected_width - small_image.width) > 1:
            big_aspect = big_image.width / big_image.height
            small_aspect = small_image.width / small_image.height
            logging.info(f'Big jpg rejected for aspect ratio: {big_path} {big_aspect:.3f} {small_aspect:.3f}')
            big_image = None
        else:
            logging.info(f'Big jpg valid: {big_path}')
    else:
        # big image was missing from the directory
        big_image = None
        logging.info(f'Big jpg missing: {small_path}')
    return small_image, big_image


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate training data",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         required=True,
                         metavar='config.json',
                         help='config file')

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
        logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
        print(f'Logfile {logfile}')
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        print(f'Log stdout')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    jpg_index = get_jpg_filenames(config['jpg_directory'],
                                  config.get('big_jpg_directory'))
    agreed_filename = config['agreed_file']
    output_directory = config['training_directory']

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

    for k, v in filters.items():
        logging.info(f'Filter: {k} {v}')

    if config.get('sampling', False):
        samples = [0.25, 0.5, 0.75, 1.0]
    else:
        samples = [1.0]

    max_size = config.get('max_size')
    holdout = config.get('holdout', 0)

    logging.info(f'Loading {input_filename}')
    big_df = pd.read_csv(input_filename)
    process_big_df(big_df, output_directory, jpg_index, max_size, samples, filters, holdout)

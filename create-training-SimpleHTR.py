#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import logging
import math
import os
import uuid
from typing import Tuple, Dict

import pandas as pd
from PIL import Image
from pandas import Series, DataFrame


def get_jpg_filenames(small_jpg_directory: str, big_jpg_directory: str) -> Dict[str, Tuple[str, str]]:
    """
    Find and index all the *.jpg files in the specified directories.
    :param big_jpg_directory:
    :param small_jpg_directory:
    :return: index from basename to tuple[2] of full paths
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


def make_workflow_index(dataframe: DataFrame, samples0: list, main_output_directory: str) -> dict:
    """
    Index and create output subdirectories for the workflows
    :param samples0: list of sampling rates (floats)
    :param dataframe: dataframe with a 'workflow' column
    :param main_output_directory:
    :return: index from workflow ID and sampling to full path
    """
    workflows = dataframe['workflow_id'].unique()
    index = dict()
    for workflow in workflows:
        for sample in samples0:
            # subdirectory, e.g., "23456_0.75" for workflow and sampling rate
            wf_directory = os.path.join(main_output_directory, f'{workflow}_{sample:.2f}')
            index[(workflow, sample)] = wf_directory
            logging.info(f'Workflow directory {wf_directory}')
            os.makedirs(wf_directory, exist_ok=True)
    return index


def process_big_df(dataframe: DataFrame, output_directory0, jpg_index0: dict,
                   stem0: str, suppress_spaces0: bool, digits_only0: bool,
                   max_size0: int, samples0: list):
    """
    Process a dataframe from our CSV format to generate training data
    :param samples0:
    :param max_size0:
    :param digits_only0:
    :param suppress_spaces0:
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

    # clean the ends of the strings (e.g., newlines)
    dataframe['text'] = dataframe['text'].apply(lambda x: x.strip())
    logging.warning(f'Stripped whitespace from ends of text')

    # TODO replace the following with custom regex for each workflow
    # optional: delete rows with anything other than digits (in case it blows up SimpleHTR)
    if digits_only0:
        dataframe = dataframe[dataframe['text'].str.fullmatch(r'\d+', na=False)]
        logging.warning(f'Dropped text with non-digits; filtered rows {dataframe.shape[0]}')

    # optional: delete rows with spaces in the text (in case it blows up SimpleHTR)
    if suppress_spaces0:
        dataframe = dataframe[dataframe['text'].str.contains(" ") == False]
        logging.warning(f'Dropped text with spaces; filtered rows {dataframe.shape[0]}')

    dataframe['x_r'] = dataframe['x'].apply(math.floor)
    dataframe['y_r'] = dataframe['y'].apply(math.floor)
    dataframe['width_r'] = dataframe['width'].apply(math.ceil)
    dataframe['height_r'] = dataframe['height'].apply(math.ceil)

    # split into separate dataframes by workflow_id
    grouped = dataframe.groupby(dataframe['workflow_id'])
    workflow_index = make_workflow_index(dataframe, samples0, output_directory0)

    for wf_id in dataframe['workflow_id'].unique():
        sub_dataframe = grouped.get_group(wf_id)
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
    img_subdirectory  = os.path.join(output_subdirectory, 'img')
    os.makedirs(img_subdirectory, exist_ok=True)

    words_file_data = []
    words_file_path = os.path.join(output_subdirectory, 'words.txt')
    filename_prefix = uuid.uuid4().hex

    # group by filename so we load each JPG file only once
    grouped_df = dataframe.groupby(dataframe['filename'])

    for filename in dataframe['filename'].unique():
        # TODO WIP HERE
        small_image, big_image = load_images(filename, jpg_index0)
        small_df = grouped_df.get_group(filename)

        for index, row in small_df.iterrows():
            cropped_image = crop_image(small_image, big_image, row)
            info_path = os.path.join(output_subdirectory, 'info', f'{stem0}_{index:09}.csv')
            png_filename = f'{filename_prefix}_{index:07d}.png'
            png_path = os.path.join(img_subdirectory, png_filename)
            words_file_data.append(generate_words_row(png_filename, row))
            save_shtr_files(cropped_image, row, info_path, png_path)

    # write the words.txt file
    with open(words_file_path, 'w') as f0:
        words_file_data.sort()
        for line in words_file_data:
            f0.write(line)
    logging.info(f'Saved words file {words_file_path}')
    logging.info(f'Number of entries: {len(words_file_data)}')
    return


def generate_words_row(png_filename, row: Series):
    """
    simplified words.txt format
    :param png_filename:
    :param row:
    :return:
    """
    # refer to SimpleHTR.md for the meaning of these
    text = row['text']
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


oparser = argparse.ArgumentParser(description="Generate training data",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str,
                     required=True,
                     metavar='FILE.json',
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
    print(f'Logfile f{logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:
    print(f'Log stdout')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

jpg_index = get_jpg_filenames(config['jpg_directory'],
                              config.get('big_jpg_directory'))
input_filename = config['intermediate_file']
output_directory = config['training_directory']

if config.get('sampling', False):
    samples = [0.25, 0.5, 0.75, 1.0]
else:
    samples = [1.0]

suppress_spaces = config.get('suppress_spaces', False)
digits_only = config.get('digits_only', False)
max_size = config.get('max_size')

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)
stem = os.path.splitext(os.path.basename(input_filename))[0]
process_big_df(big_df, output_directory, jpg_index, stem, suppress_spaces, digits_only, max_size, samples)

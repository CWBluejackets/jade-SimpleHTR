#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import logging
import math
import os

import pandas as pd
from PIL import Image
from pandas import Series, DataFrame


def get_jpg_filenames(directory: str) -> dict:
    """
    Find and index all the *.jpg files in the specified directory.
    :param directory:
    :return: index from basename to full path
    """
    pattern = os.path.join(directory, '**', '*.jpg')
    logging.info(f'Using {pattern}')
    jpg_glob = glob.iglob(pattern, recursive=True)
    index = {os.path.basename(x): x for x in jpg_glob}
    logging.info(f'Found {len(index)} jpg files')
    return index


def make_workflow_index(dataframe: DataFrame, directory: str):
    """
    Index and create output subdirectories for the workflows
    :param dataframe: dataframe with a 'workflow' column
    :param directory: index from workflow number to full path
    :return:
    """
    workflows = dataframe['workflow_id'].unique()
    index = {workflow: os.path.join(directory, str(workflow)) for workflow in workflows}
    for wf_directory in index.values():
        logging.info(f'Workflow directory {wf_directory}')
        os.makedirs(wf_directory, exist_ok=True)
    return index


def process_big_df(dataframe: DataFrame, output_directory0, jpg_index0: dict,
                   stem0: str, suppress_spaces0: bool):
    """
    Process a dataframe from our CSV format to generate training data
    :param output_directory0:
    :param dataframe:
    :param jpg_index0: the JPG file index
    :param stem0: starter for the output filenames
    :return:
    """
    logging.warning(f'Unfiltered rows {dataframe.shape[0]}')

    # delete the rows without rectangles
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text'])
    logging.warning(f'Dropped NaN rectangles; filtered rows {dataframe.shape[0]}')

    # delete rows with jpg filenames that are not available
    dataframe = dataframe[dataframe['filename'].isin(jpg_index0.keys())]
    logging.warning(f'Dropped unavailable jpgs; filtered rows {dataframe.shape[0]}')

    # optional: delete rows with spaces in the text (in case it blows up SimpleHTR)
    if suppress_spaces0:
        dataframe = dataframe[dataframe["text"].str.contains(" ") == False]
        logging.warning(f'Dropped text with spaces; filtered rows {dataframe.shape[0]}')

    # round x and y down; width and height up
    dataframe['x_r'] = dataframe['x'].apply(math.floor)
    dataframe['y_r'] = dataframe['y'].apply(math.floor)
    dataframe['width_r'] = dataframe['width'].apply(math.ceil)
    dataframe['height_r'] = dataframe['height'].apply(math.ceil)

    # split into separate dataframes by workflow_id
    grouped = dataframe.groupby(dataframe['workflow_id'])
    workflow_index = make_workflow_index(dataframe, output_directory0)

    for wf_id in dataframe['workflow_id'].unique():
        wf_df = grouped.get_group(wf_id)
        workflow_directory = workflow_index[wf_id]
        logging.info(f'Starting workflow {wf_id} with {wf_df.shape[0]} rows')
        process_workflow_df(wf_df, workflow_directory, jpg_index0, stem0)
    return


def process_workflow_df(dataframe: DataFrame, output_subdirectory: str, jpg_index0: dict, stem0: str):
    os.makedirs(output_subdirectory, exist_ok=True)
    info_subdirectory = os.path.join(output_subdirectory, 'info')
    os.makedirs(info_subdirectory, exist_ok=True)
    gt_subdirectory = os.path.join(output_subdirectory, 'gt')
    os.makedirs(gt_subdirectory, exist_ok=True)

    words_file_data = []
    words_file_path = os.path.join(gt_subdirectory, 'words.txt')

    # group by filename so we load each file only once
    grouped_df = dataframe.groupby(dataframe['filename'])

    for filename in dataframe['filename'].unique():
        current_image = load_image(filename, jpg_index0)
        small_df = grouped_df.get_group(filename)

        for index, row in  small_df.iterrows():
            cropped_image = crop_image(current_image, row)
            info_filename = os.path.join(output_subdirectory, 'info', f'{stem0}_{index:09}.csv')
            subdirectory, png_file, png_path = make_png_dir_file(index, output_subdirectory)
            words_file_data.append(generate_words_row(png_file, row))
            os.makedirs(subdirectory, exist_ok=True)
            save_shtr_files(cropped_image, row, info_filename, png_path)

    # write the words.txt file
    with open(words_file_path, 'w') as f0:
        for line in words_file_data:
            f0.write(line)
    logging.info(f'Saved words file {words_file_path}')
    return


def generate_words_row(png_filename, row: Series):
    """
    Refer to SimpleHTR.md for the format
    :param png_filename:
    :param row:
    :return:
    """
    # refer to SimpleHTR.md for the meaning of these
    data = [png_filename, 'ok', 154,
            row['x_r'], row['y_r'], row['width_r'], row['height_r'],
            'CD', row['text'].strip()]
    # strip() added because apparently some have newlines ?!
    line = ' '.join([str(item) for item in data]) + '\n'
    return line


def make_png_dir_file(index: int, wf_directory: str):
    """
    Refer to SimpleHTR.md for the format
    :param wf_directory: directory for the workflow output
    :param index: row number from the main dataframe, used for uniqueness
    :return: path and filename
    """
    # img/a01/a01-000u/a01-000u-00-00.png
    s = f'{index:09d}'
    a = s[0:2]
    b = s[2:5]
    c = s[5:7]
    d = s[7:9]
    directory = os.path.join(wf_directory, 'img', f'a{a}', f'a{a}-{b}u')
    filename = f'a{a}-{b}-{c}-{d}.png'
    fullpath = os.path.join(directory, filename)
    return directory, filename, fullpath


def crop_image(image: Image, row: Series) -> Image:
    """
    Crop the image according to the coordinates in the annotation.
    :param image:
    :param row:
    :return: cropped image
    """
    # PIL: (0, 0) is top left
    # same as Zooniverse and x, y = top left corner of box
    # PIL box: tuple (left, upper, right, lower)
    left = row['x']
    top = row['y']
    right = left + row['width']
    bottom = top + row['height']
    cropped = image.crop(box=(left, top, right, bottom))
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
    row['png_filename'] = png_path
    row.to_csv(info_file_path)
    logging.info(f'Saved info {info_file_path}')

    cropped_image.save(png_path, bits=8)
    logging.info(f'Saved image {png_path}')
    return


def load_image(basename: str, jpg_index0) -> Image:
    """
    Locate a jpg file from its basename and load it
    :param basename: 'foo.jpg'
    :param jpg_index0: the index
    :return: PIL image
    """
    fullpath = jpg_index0[basename]
    image = Image.open(fullpath, 'r')
    return image


oparser = argparse.ArgumentParser(description="Generate training data",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     required=True, metavar='FILE.json',
                     help='config file')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

# turn off the chained assignment warnings
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

timestamp = datetime.datetime.now().isoformat(timespec='seconds')
logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
print('Logfile', logfile)
logging.basicConfig(filename=logfile, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

jpg_index = get_jpg_filenames(config['jpg_directory'])
input_filename = config['intermediate_file']
output_directory = config['training_directory']

suppress_spaces = config.get('suppress_spaces', False)

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)
stem = os.path.splitext(os.path.basename(input_filename))[0]
process_big_df(big_df, output_directory, jpg_index, stem, suppress_spaces)

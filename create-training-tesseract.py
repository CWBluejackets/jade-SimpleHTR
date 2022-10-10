#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import logging
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
    workflows = set(dataframe['workflow_id'])
    index = {workflow: os.path.join(directory, str(workflow)) for workflow in workflows}
    for wf_directory in index.values():
        logging.info(f'Workflow directory {wf_directory}')
        os.makedirs(wf_directory, exist_ok=True)
    return index


def process_df(dataframe: DataFrame, output_directory0, jpg_index0: dict, stem0: str):
    """
    Process a dataframe from our CSV format to generate Tesseract training data
    :param output_directory0:
    :param dataframe:
    :param jpg_index0: the JPG file index
    :param stem0: starter for the output filenames
    :return:
    """
    logging.warning(f'Unfiltered rows {dataframe.shape[0]}')

    # delete the rows without rectangles
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text'])
    logging.warning(f'Filtered rows {dataframe.shape[0]}')

    # delete rows with jpg filenames that are not available
    dataframe = dataframe[df['filename'].isin(jpg_index0.keys())]
    logging.warning(f'Filtered rows {dataframe.shape[0]}')

    # sort by filename for efficiency (to load each jpg once only)
    dataframe = dataframe.sort_values(by='filename', axis=0)
    workflow_index = make_workflow_index(dataframe, output_directory0)

    current_filename = None
    current_image = None
    for index, row in dataframe.iterrows():
        filename = row['filename']
        if filename != current_filename:
            current_image = load_image(filename, jpg_index0)
            current_filename = filename
        output_subdirectory = workflow_index[row['workflow_id']]
        cropped_image = crop_image(current_image, row)
        output_base = os.path.join(output_subdirectory, f'{stem0}_{index:010}')
        save_tesseract_files(output_base, cropped_image, row)
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


def save_tesseract_files(output_base: str, cropped_image: Image, row: Series):
    """
    Save all the files
    :param output_base: directory and start of filename
    :param cropped_image:
    :param row:
    :return:
    """
    info_filename = f'{output_base}.csv'
    row.to_csv(info_filename)
    logging.info(f'Saved {info_filename}')

    tif_filename = f'{output_base}.tif'
    # TODO check the dpi (copied from the old meme generator)
    cropped_image.save(tif_filename, dpi=(70, 70))
    logging.info(f'Saved {tif_filename}')

    text = row['text']
    text_filename = f'{output_base}.raw.txt'
    with open(text_filename, 'w') as f0:
        f0.write(text)
    logging.info(f'Saved {text_filename}')

    box_filename = f'{output_base}.box'
    save_box(box_filename, row['width'], row['height'], text)
    return


def save_box(filename: str, width: int, height: int, text: str):
    """
    Save the tesseract box file.
    :param filename:
    :param width:
    :param height:
    :param text:
    :return:
    """
    # tesseract box format: left bottom right top page
    # Use page=0 for single-page images.
    # Note that (0,0) is the BOTTOM left!
    with open(filename, 'w') as f0:
        left = 0
        right = width
        top = height
        bottom = 0
        line0 = 'WordStr %d %d %d %d 0 #%s\n' % (left, bottom, right, top, text)
        f0.write(line0)
        line1 = '\t %d %d %d %d 0\n' % (left, bottom, right, top)
        f0.write(line1)
    logging.info(f'Saved {filename}')
    return


oparser = argparse.ArgumentParser(description="Generate training data",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     required=True, metavar='FILE.json',
                     help='config file')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

timestamp = datetime.datetime.now().isoformat(timespec='seconds')
logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
print('Logfile', logfile)
logging.basicConfig(filename=logfile, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

jpg_index = get_jpg_filenames(config['jpg_directory'])
input_filename = config['intermediate_file']
output_directory = config['training_directory']

logging.info(f'Loading {input_filename}')
df = pd.read_csv(input_filename)
stem = os.path.splitext(os.path.basename(input_filename))[0]
process_df(df, output_directory, jpg_index, stem)

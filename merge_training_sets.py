#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil

INITIAL_A = re.compile(r'^a')


def output_letter(i):
    """
    0a, 1b, 2c, etc.
    :param i:
    :return:
    """
    return chr(ord('a') + i)


def update_filename(filename, i):
    """
    update the 'a' at the beginning of the filename
    """
    if i == 0:
        return filename
    letter = output_letter(i)
    return INITIAL_A.sub(letter, filename, 1)


oparser = argparse.ArgumentParser(description="Merge training data sets",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str,
                     required=True,
                     metavar='FILE.json',
                     help='config file')

oparser.add_argument('-o', dest='output_name',
                     type=str,
                     required=True,
                     metavar='NNNNN',
                     help='subdirectory for merged output')

oparser.add_argument('input_names',
                     nargs='+',
                     metavar='SUBDIRS',
                     help='input training data subdirectories')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

training_directory = config['training_directory']
output_directory = os.path.join(training_directory, options.output_name)
os.makedirs(output_directory, exist_ok=True)
os.makedirs(os.path.join(output_directory, 'gt'), exist_ok=True)
output_img_dir = os.path.join(output_directory, 'img')
os.makedirs(output_img_dir, exist_ok=True)

output_words = []

for counter, input_name in enumerate(options.input_names):
    input_directory = os.path.join(training_directory, input_name)
    with open(os.path.join(input_directory, 'gt', 'words.txt'), 'r') as f:
        input_words = [line.strip() for line in f.readlines()]
    for line in input_words:
        output_words.append(update_filename(line, counter))
    img_directory = os.path.join(input_directory, 'img')
    img_glob = glob.iglob(os.path.join(img_directory, '*', '*', '*.png'))
    for img_path in img_glob:
        img_relpath = os.path.relpath(img_path, img_directory)
        part_a, part_b, img_filename = img_relpath.split('/')
        output_img_subdir = os.path.join(output_img_dir,
                                         update_filename(part_a, counter),
                                         update_filename(part_b, counter))
        os.makedirs(output_img_subdir, exist_ok=True)
        new_img_path = os.path.join(output_img_subdir, update_filename(img_filename, counter))
        shutil.copyfile(img_path, new_img_path)

with open(os.path.join(output_directory, 'gt', 'words.txt'), 'w') as f:
    for line in output_words:
        f.write(f'{line}\n')

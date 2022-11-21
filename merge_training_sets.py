#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil


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
output_img_dir = os.path.join(output_directory, 'img')
os.makedirs(output_img_dir, exist_ok=True)

output_words = []

for counter, input_name in enumerate(options.input_names):
    input_directory = os.path.join(training_directory, input_name)
    input_words_file = os.path.join(input_directory, 'words.txt')
    with open(input_words_file, 'r') as f:
        input_words = f.readlines()
        print(f'Read {len(input_words)} lines from {input_words_file}')
    output_words.extend(input_words)

    img_directory = os.path.join(input_directory, 'img')
    img_glob = glob.glob(os.path.join(img_directory, '*.png'))
    print(f'Copying {len(img_glob)} files from {img_directory} to {output_img_dir}')

    for img_path in img_glob:
        img_filename = os.path.basename(img_path)
        new_img_path = os.path.join(output_img_dir, img_filename)
        shutil.copyfile(img_path, new_img_path)

output_words_file = os.path.join(output_directory, 'words.txt')

with open(output_words_file, 'w') as f:
    print(f'Writing {len(output_words)} lines to {output_words_file}')
    for line in output_words:
        f.write(f'{line}')

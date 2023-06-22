#!/usr/bin/env python3
import argparse
from collections import Counter


def load_file(control_file: str) -> dict:
    result = dict()
    with open(control_file) as f:
        for line in f:
            # ignore empty and comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # maxsplit is max number of cuts, not number of pieces
            line_split = line.split(' ', maxsplit=1)
            # columns: filename, label
            image_file = line_split[0]
            if image_file in result:
                print(f'Warning: duplicate filename {image_file} in {control_file}')
            # sometimes the training data includes a correct empty string
            if len(line_split) == 2:
                label = line_split[1]
            else:
                label = ' '
            result[image_file] = label
    return result


def get_label_count(control_dict: dict) -> Counter:
    return Counter(control_dict.values())


oparser = argparse.ArgumentParser(description="Compare labels in two control files",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument(dest='filenames',
                     nargs=2,
                     required=True, metavar='TRAINING HOLDOUT',
                     help='control files to compare')

options = oparser.parse_args()

training_data = load_file(options.filenames[0])
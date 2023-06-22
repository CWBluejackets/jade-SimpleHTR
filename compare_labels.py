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


def sort_set_by_counter(input_set: set, input_counter: Counter) -> list:
    result = [(item, input_counter[item]) for item in input_set]
    result.sort(key=lambda item: - item[1])
    return result


oparser = argparse.ArgumentParser(description="Compare labels in two control files",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument(dest='filenames',
                     nargs=2,
                     metavar='TRAINING.txt HOLDOUT.txt',
                     help='control files to compare')

options = oparser.parse_args()
training_file, holdout_file = options.filenames

training_data = load_file(training_file)
holdout_data = load_file(holdout_file)

training_labels = get_label_count(training_data)
holdout_labels = get_label_count(holdout_data)

print(f'LABELS ONLY IN {training_file}')
label_counts = sort_set_by_counter(training_labels.keys() - holdout_labels.keys(),
                                   training_labels)
for label in label_counts:
    print(f'{label[0]:35s} {label[1]:5d}')

print()
print(f'LABELS ONLY IN {holdout_file}')
label_counts = sort_set_by_counter(holdout_labels.keys() - training_labels.keys(),
                                   holdout_labels)
for label in label_counts:
    print(f'{label[0]:35s} {label[1]:5d}')

print()
print(f'LABELS IN BOTH FILES')
label_counts = sort_set_by_counter(training_labels.keys() & holdout_labels.keys(),
                                   training_labels)
for label in label_counts:
    print(f'{label[0]:35s} {label[1]:5d} {holdout_labels[label[0]]:5d}')

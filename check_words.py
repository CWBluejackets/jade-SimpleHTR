#!/usr/bin/env python3
import argparse
from collections import Counter


# splitting in dataloader:
# split_idx = int(data_split * len(self.samples))
# self.train_samples = self.samples[:split_idx]
# self.validation_samples = self.samples[split_idx:]


def process(filename0, options0):
    print()
    print(f'Reading {filename0}')
    with open(filename0, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    nbr = len(lines)
    cut = int(options0.training * nbr)
    print(f'Number of entries: {nbr}')
    words = [line.split(' ', maxsplit=1)[1] for line in lines]
    top = words[:cut]
    bottom = words[cut:]
    print(f'Training size: {len(top)}')
    print(f'Testing size: {len(bottom)}')
    top_count = Counter(top)
    bot_count = Counter(bottom)
    print(f'Training: unique labels: {len(top_count)}')
    print(f'Testing: unique labels: {len(bot_count)}')
    shared = top_count & bot_count
    print(f'Nbr of labels in common: {len(shared)}')
    for label in shared:
        print(f'Label {label}')
        print(f'  training instances: {top_count[label]}')
        print(f'  testing instances: {bot_count[label]}')
    return


parser = argparse.ArgumentParser(description="check labels in words___.txt for overlaps",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-t', dest='training',
                    type=float,
                    default=0.95,
                    metavar='N',
                    help='training split')

parser.add_argument('input_files', metavar='FILE', nargs='*',
                    type=str,
                    help='csv files')

options = parser.parse_args()

for filename in options.input_files:
    process(filename, options)


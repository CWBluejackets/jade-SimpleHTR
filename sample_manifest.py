#!/usr/bin/env python3
import argparse
import json
import random

if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="Generate Zooniverse manifest")

    oparser.add_argument('-o', dest='output_file',
                         type=str,
                         metavar='FILE.json',
                         help='output file')

    oparser.add_argument('-n', dest='sample_size',
                         type=int,
                         metavar='INT',
                         help='number of entries to save in output')

    oparser.add_argument(dest='input_files',
                         nargs='+',
                         metavar='FILE[S]',
                         help='input file(s)')

    options = oparser.parse_args()

    if options.sample_size and not options.output_file:
        raise ValueError("option -n requires option -o")

    input_data = []

    for input_file in options.input_files:
        print(f'Reading {input_file}')
        with open(input_file, 'r') as f:
            batch = json.load(f)
            print(f'Loaded {len(batch)} entries')
            input_data.extend(batch)

    print(f'Total input size {len(input_data)}')

    if options.output_file:
        if options.sample_size:
            output_data = random.sample(input_data, options.sample_size)
        else:
            output_data = input_data
        print(f'Total output size {len(output_data)}')
        print(f'Writing {options.output_file}')
        with open(options.output_file, 'w') as f:
            json.dump(output_data, f, indent=1, sort_keys=True)


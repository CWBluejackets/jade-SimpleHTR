#!/usr/bin/env python3
import argparse
import os.path

import Levenshtein
import pandas as pd

# Script to read test csv output and produce
# character error, word accuracy, and other statistics


def is_label_correct(row: pd.Series) -> bool:
    return row['label'] == row['output']


def get_distance(row: pd.Series) -> int:
    return Levenshtein.distance(str(row['label']), str(row['output']))


def process_csv(input0, output0):
    results = []
    dataframe = pd.read_csv(input0, index_col=0, header=0,
                            dtype={'label': str, 'output': str})
    # generate statistics and overall scores
    pearson = dataframe[['similarity', 'probability', 'logprob', 'distance', 'length']].corr(method='pearson')
    results.extend(['PEARSON CORRELATIONS', pearson.to_string(), ''])
    nbr_correct = dataframe['correct'].value_counts()[True]
    word_accuracy = nbr_correct / dataframe['correct'].size
    results.extend(['WORD ACCURACY', f'{word_accuracy}', f'{word_accuracy:.1%}', ''])
    character_error = dataframe['distance'].sum() / dataframe['length'].sum()
    results.extend(['CHARACTER ERROR', f'{character_error}', f'{character_error:.1%}', ''])
    dump(results, output0)
    return


def dump(results, output0):
    if output0:
        with open(output0, 'w') as f:
            for line in results:
                f.write(f'{line}\n')
    else:
        for line in results:
            print(f'{line}')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse test CSV",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_files', metavar='FILE(s)', nargs='+',
                        help='input files')

    parser.add_argument('-o', dest='output_directory',
                        type=str,
                        default=None,
                        metavar='FILE',
                        help='output file')

    options = parser.parse_args()

    for input_file in options.input_files:
        if options.output_directory:
            output_file = os.path.splitext(os.path.basename(input_file))[0] + '_scores.txt'
            output_path = os.path.join(os.path.dirname(options.output_directory),
                                       output_file)
        else:
            output_path = None
        process_csv(input_file, output_path)

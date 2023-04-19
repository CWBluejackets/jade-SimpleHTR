#!/usr/bin/env python3
import argparse
import os.path
from io import BytesIO

import Levenshtein
import pandas as pd
import matplotlib.pyplot as plt

# Script to read test csv output and produce
# character error, word accuracy, and other statistics


def is_label_correct(row: pd.Series) -> bool:
    return row['label'] == row['output']


def get_distance(row: pd.Series) -> int:
    return Levenshtein.distance(str(row['label']), str(row['output']))


def process_csv(dataframe0):
    results0 = []
    # generate statistics and overall scores
    pearson = dataframe0[['similarity', 'probability', 'logprob', 'distance', 'length']].corr(method='pearson')
    results0.extend(['PEARSON CORRELATIONS', pearson.to_string(), ''])
    nbr_correct = dataframe0['correct'].value_counts()[True]
    word_accuracy = nbr_correct / dataframe0['correct'].size
    results0.extend(['WORD ACCURACY', f'{word_accuracy}', f'{word_accuracy:.1%}', ''])
    character_error = dataframe0['distance'].sum() / dataframe0['length'].sum()
    results0.extend(['CHARACTER ERROR', f'{character_error}', f'{character_error:.1%}', ''])
    return results0


def generate_scatter_plots(dataframe0: pd.DataFrame, output0):
    combinations = [('probability', 'similarity'), ('logprob', 'similarity'),
                    ('probability', 'distance'), ('length', 'similarity')]
    for i, columns in enumerate(combinations):
        buffer = BytesIO()
        ax = dataframe0.plot.scatter(*columns)
        ax.get_figure()
        filename = f'{output0}_{i:02d}.png'
        plt.savefig(filename, dpi=200, format='png')
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
        dataframe = pd.read_csv(input_file, index_col=0, header=0,
                                dtype={'label': str, 'output': str})
        results = process_csv(dataframe)
        if options.output_directory:
            output_path = os.path.join(os.path.dirname(options.output_directory),
                                       os.path.splitext(os.path.basename(input_file))[0])
            dump(results, output_path + '_scores.txt')
            generate_scatter_plots(dataframe, output_path)
        else:
            dump(results, None)



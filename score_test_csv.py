#!/usr/bin/env python3
import argparse
import os.path
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt

# Script to read test csv output and produce
# character error, word accuracy, and other statistics


COMBINATIONS = [('probability', 'similarity'), ('logprob', 'similarity'),
                ('probability', 'distance'), ('length', 'probability'),
                ('length', 'similarity'),
                ('length', 'distance')]


def is_label_correct(row: pd.Series) -> bool:
    return row['label'] == row['output']


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
    for i, columns in enumerate(COMBINATIONS):
        ax = dataframe0.plot.scatter(*columns)
        fig = ax.get_figure()
        filename = f'{output0}_{i:02d}.png'
        print(f'saving to {filename}')
        plt.savefig(filename, dpi=200, format='png')
        # https://stackoverflow.com/questions/45933886/python-plt-close-or-clear-figure-does-not-work
        plt.close(fig)
    return


def dump(results0, output0):
    if output0:
        print(f'saving to {output0}')
        with open(output0, 'w') as f:
            for line in results0:
                f.write(f'{line}\n')
    else:
        for line in results0:
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
            os.makedirs(options.output_directory, exist_ok=True)
            output_path = os.path.join(options.output_directory,
                                       os.path.splitext(os.path.basename(input_file))[0])
            dump(results, output_path + '_scores.txt')
            generate_scatter_plots(dataframe, output_path)
        else:
            dump(results, None)



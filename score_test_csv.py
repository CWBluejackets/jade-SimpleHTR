#!/usr/bin/env python3
import argparse
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Script to read test csv output and produce
# character error, word accuracy, and other statistics


COMBINATIONS = [('probability', 'similarity'),
                ('logprob', 'similarity'),
                ('distance', 'probability'),
                ('distance', 'similarity'),
                ('length', 'probability'),
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


def generate_heatmap(dataframe0: pd.DataFrame, output0: str, cuts: int):
    crosstab = pd.crosstab(pd.cut(dataframe0['similarity'], cuts), pd.cut(dataframe0['probability'], cuts))
    ax = sns.heatmap(crosstab, annot=True)
    fig = ax.get_figure()
    filename = f'{output0}_heatmap.png'
    print(f'saving to {filename}')
    plt.savefig(filename, dpi=200, format='png')
    # https://stackoverflow.com/questions/45933886/python-plt-close-or-clear-figure-does-not-work
    plt.close(fig)
    return


def describe_cutoff(df0: pd.DataFrame, cutoff: float):
    original_length = df0['similarity'].count()
    df1 = df0[df0['probability'] >= cutoff]
    rel_size = df1['similarity'].count()
    return rel_size / original_length, df1['similarity'].min(), df1['similarity'].median(), \
        df1['similarity'].mean(), df1['similarity'].max()


def generate_cutoff_table(dataframe0: pd.DataFrame, nbr_cuts: int) -> pd.DataFrame:
    bottom = dataframe0['probability'].min()
    top = dataframe0['probability'].max()
    step = (top - bottom) / nbr_cuts
    cuts = [bottom + i * step for i in range(nbr_cuts)]
    cutoff_array = [describe_cutoff(dataframe0, c) for c in cuts]
    return pd.DataFrame(data=cutoff_array, index=cuts,
                        columns=['retained', 'min_sim', 'med_sim', 'mean_sim', 'max_sim'])


def generate_cutoff_plot(dataframe0: pd.DataFrame, output0, nbr_cuts: int):
    cut_table = generate_cutoff_table(dataframe0, nbr_cuts)
    ax = cut_table.plot.line()
    fig = ax.get_figure()
    filename = f'{output0}_cutoffs.png'
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
                        metavar='DIR',
                        help='output directory')

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
            generate_heatmap(dataframe, output_path, 20)
            generate_cutoff_plot(dataframe, output_path, 50)
        else:
            dump(results, None)

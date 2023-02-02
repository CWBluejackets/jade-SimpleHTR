#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import re
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import Levenshtein

WHITESPACE = re.compile(r'\s+')
PIXEL_CUTOFF = 5.0
VERTICAL_OVERLAP_THRESHOLD = 0.6


def flag_wrong_column_row(row: pd.Series, wrong_combos: list) -> bool:
    key = (row['subject_id'], row['workflow_id'], row['annotator'])
    return key in wrong_combos


def detect_wrong_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    # New version: group by subject-workflow, then go through each combination to eliminate wrong-column annotators.
    # NOTE: deletion of an annotator applies to this subject-workflow combination only.
    wrong_column_combos = []
    # contains (subject, workflow, annotator) triples

    grouped_columns = dataframe.groupby(by=['subject_id', 'workflow_id'])
    for muster_key in grouped_columns.groups.keys():
        muster_column_df = grouped_columns.get_group(muster_key)
        nbr_annotators = muster_column_df['annotator'].unique().size
        # You can't calculate overlaps with 1 annotator.
        # If there are 2 and they don't overlap, you can't tell which is right.
        if nbr_annotators > 2:
            annotator_ranges = dict()
            annotator_grouped = muster_column_df.groupby('annotator')
            for annotator in annotator_grouped.groups.keys():
                anno_df = annotator_grouped.get_group(annotator)
                annotator_ranges[annotator] = (anno_df['left'].min(), anno_df['right'].max())
            annotator_set = set(annotator_ranges.keys())
            for annotator, (left, right) in annotator_ranges.items():
                others_left = min([annotator_ranges[x][0] for x in annotator_set - {annotator}])
                others_right = max([annotator_ranges[x][1] for x in annotator_set - {annotator}])
                overlap = get_overlap_proportion(left, right, others_left, others_right)
                if overlap == 0.0:
                    wrong_column_combos.append((muster_column_df[0], muster_column_df[1], annotator))

    # mark all the matching lines
    dataframe['wrong_column'] = dataframe.apply(lambda row0: flag_wrong_column_row(row0, wrong_column_combos),
                                                axis=0)
    return dataframe


def basic_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Delete the rows with missing rectangles, cleaned text, or required identifiers
    :param dataframe:
    :return:
    """
    logging.warning(f'Starting rows {dataframe.shape[0]}')
    # Note: the text_clean column already contains a single space where the exported text is blank
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['left', 'right', 'top', 'bottom', 'text_clean',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text/IDs; filtered rows {dataframe.shape[0]}')

    # create a column with a special value to start it as int64
    # dataframe['line'] = -1
    # this would break combine_first(...) later?
    return dataframe


def series_levenshtein(text_series: pd.Series):
    '''
    Generate a column of Levenshtein similarity ratios
    :param text_series: column of strings
    :return: column giving the similarity between each string and the previous one
    '''
    texts = text_series.values
    output = [np.NAN] + [Levenshtein.ratio(texts[i - 1], texts[i]) for i in range(1, texts.shape[0])]
    return pd.Series(output, index=text_series.index)


def generate_line_series(overlap_series: pd.Series) -> pd.Series:
    '''
    Generate line numbers for one muster sheet (subject-workflow combination)
    :param overlap_series: column of vertical overlaps
    :return: column of line numbers
    '''
    line = 0
    results = [line]
    # value 0 doesn't count
    for this_overlap in overlap_series.values[1:]:
        if this_overlap < VERTICAL_OVERLAP_THRESHOLD:
            line += 1
        results.append(line)
    return pd.Series(results, index=overlap_series.index)


def generate_line_numbers(dataframe: pd.DataFrame) -> pd.DataFrame:
    # split by subject and workflow ID then sort vertically
    muster_grouping = dataframe.groupby(by=['subject_id', 'workflow_id'])
    for muster_key in muster_grouping.groups.keys():
        muster_sheet = muster_grouping.get_group(muster_key)
        # This way of sorting produces a cleaner separation in the histogram (empirically)
        muster_sheet = muster_sheet.sort_values(by=['top', 'bottom'], ascending=[True, False])

        muster_sheet['prev_top'] = muster_sheet['top'].rolling(window=2).min()
        muster_sheet['prev_bottom'] = muster_sheet['bottom'].rolling(window=2).min()
        muster_sheet['levenshtein'] = series_levenshtein(muster_sheet['text_clean'])
        muster_sheet['v_overlap'] = muster_sheet.apply(
            lambda row: get_overlap_proportion(row['top'], row['bottom'], row['prev_top'], row['prev_bottom']), axis=1)
        muster_sheet['line'] = generate_line_series(muster_sheet['v_overlap'])
        # Copy the new columns back to the main dataframe
        dataframe = dataframe.combine_first(
            muster_sheet[['prev_top', 'prev_bottom', 'levenshtein', 'v_overlap', 'line']])
    return dataframe


def process_big_df(dataframe):
    """
    Analyse the input file, generate line numbers, and produce a quasi-database of the annotations
    :param dataframe: the big CSV unpacked from Zooniverse
    :return: nested dict  tuple(subject, workflow, line) -> annotator -> tuple(annotation, x, width)
    """
    dataframe = basic_cleaning(dataframe)
    # TODO detect column jumping
    dataframe = detect_wrong_columns(dataframe)
    dataframe = generate_line_numbers(dataframe)
    dataframe = vote_line_by_line(dataframe)
    return dataframe


def vote_line_by_line(dataframe: pd.DataFrame) -> pd.DataFrame:
    # decision type, number of votes -> nbr of matching decisions
    decision_counter = defaultdict(Counter)

    # Process each muster sheet (subject, workflow) in line number order
    muster_grouping = dataframe.groupby(by=['subject_id', 'workflow_id'])
    for muster_key in muster_grouping.groups.keys():
        muster_sheet = muster_grouping.get_group(muster_key)
        logging.info(f'Processing {muster_key} with {muster_sheet.shape[0]} rows')
        # Drop rows with wrong column
        muster_sheet = muster_sheet[~ muster_sheet['wrong_column']]
        # Group by line
        line_grouping = muster_sheet.groupby(by=['line'])
        for line in line_grouping.groups.keys():
            muster_line = line_grouping.get_group(line)
            muster_line = decide_line(muster_line, decision_counter)
            # Copy the new columns back to the main dataframe
            dataframe = dataframe.combine_first(
                muster_line[['decision', 'decision_type', 'mode_votes', 'total_votes']])

    return dataframe


def decide_line(muster_line: pd.DataFrame, decision_counter: dict) -> pd.DataFrame:
    """
    Carry out the voting on one subject/workflow/line combination
    :param muster_line: one subject-workflow-line combination
    :param decision_counter: to keep track of decisions by type and size
    :return:
    """
    # This comes out automatically in descending order
    vote_df = muster_line['text_clean'].value_counts()
    mode = vote_df.index[0]
    mode_votes = vote_df.values[0]
    total_votes = vote_df.values.sum()

    if mode_votes == total_votes:
        decision_type = 'unanimous'
    elif mode_votes > total_votes / 2:
        decision_type = 'majority'
    elif mode_votes > vote_df.values[1]:
        decision_type = 'plurality'
    else:
        decision_type = 'failed'

    # TODO consider using Levenshtein.median(...)

    # log and apply the result
    if decision_type != 'unanimous':
        logging.info(f'Voting {decision_type}: {key_triple} has {vote_df.to_dict()}')
    decision_counter[decision_type][total_votes] += 1
    muster_line['decision'] = None if decision_type == 'failed' else mode
    muster_line['decision_type'] = decision_type
    muster_line['mode_votes'] = mode_votes
    muster_line['total_votes'] = total_votes
    return muster_line


def get_horizontal_boundaries(annotation_dict: dict):
    """
    Calculate the left and right coordinates for each annotator
    :param annotation_dict: annotator -> (annotation, x, width)
    :return: annotator -> (x_left, x_right)
    """
    # The min 1 is there to prevent division by zero when the annotator
    # created a zero-width rectangle.
    boundary_dict = {annotator: (value[1], value[1] + max(value[2], 1))
                     for annotator, value in annotation_dict.items()}
    return boundary_dict


def get_absolute_overlap(left0, right0, left1, right1):
    """
    Calculate the overlap in pixels between two horizontal ranges
    :param left0:
    :param right0:
    :param left1:
    :param right1:
    :return:
    """
    return max(0, min(right0, right1) - max(left0, left1))


def get_overlap_proportion(this_left, this_right, reference_left, reference_right):
    """
    Calculate the proportion of the "reference" width overlapped by "this" one
    :param this_left:
    :param this_right:
    :param reference_left:
    :param reference_right:
    :return:
    """
    overlap = get_absolute_overlap(this_left, this_right, reference_left, reference_right)
    return overlap / (this_right - this_left)


if __name__ == "__main__":
    oparser = argparse.ArgumentParser(description="IAA and stuff",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    oparser.add_argument('-c', dest='config_file',
                         type=str,
                         required=True,
                         metavar='config.json',
                         help='config file')

    oparser.add_argument('-d', dest='debug',
                         default=False,
                         action='store_true',
                         help='debug')

    options = oparser.parse_args()

    with open(options.config_file) as f:
        config = json.load(f)

    # turn off the chained assignment warnings
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    # TODO find the right way to isolate columns
    pd.options.mode.chained_assignment = None

    log_directory = config.get('log_directory')

    if log_directory:
        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        logfile = os.path.join(config['log_directory'], f'{timestamp}_adjudicate.log')
        print(f'Logfile {logfile}')
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        print(f'Log stdout')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    input_filename = config['intermediate_file']
    output_filename = config['agreed_file']

    logging.info(f'Loading {input_filename}')
    big_df = pd.read_csv(input_filename)
    big_df = process_big_df(big_df)

    # sort the output to make it easy to inspect the voting and overlaps
    big_df.sort_values(by=['workflow_id', 'subject_id', 'line', 'annotator'],
                       ascending=[True, True, True, True],
                       na_position='last',
                       inplace=True)

    logging.warning(f'Writing {output_filename}')
    big_df.to_csv(output_filename)

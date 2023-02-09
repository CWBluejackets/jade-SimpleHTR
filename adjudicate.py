#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import time
from collections import defaultdict, Counter
from datetime import timedelta

import Levenshtein
import numpy as np
import pandas as pd

# Minimum size for a box to be usable
PIXEL_CUTOFF = 5.0

# Line separation (determined empirically from the histogram)
VERTICAL_OVERLAP_THRESHOLD = 0.6

# Wrong column detection
TWOFER_MIN_OVERLAP = 0.7
MORE_MIN_OVERLAP = 0.1


def detect_wrong_columns(muster_sheet: pd.DataFrame):
    """
    Detect annotators working in the wrong column
    :param muster_sheet: one subject-workflow combination
    :return: same with extra columns
    """
    wrong_annotators = set()
    nbr_annotators = muster_sheet['annotator'].unique().size

    # You can't calculate overlaps with 1 annotator.
    # If there are two columns with poor overlap, we assume one of them is wrong
    # but can't decide which so we mark them both for exclusion.
    if nbr_annotators > 1:
        annotator_ranges = dict()
        annotator_grouped = muster_sheet.groupby('annotator')
        for annotator in annotator_grouped.groups.keys():
            anno_df = annotator_grouped.get_group(annotator)
            annotator_ranges[annotator] = (anno_df['left'].min(), anno_df['right'].max())
        annotator_set = set(annotator_ranges.keys())
        lowest_overlap = 1.0
        for annotator, (left, right) in annotator_ranges.items():
            others_left = min([annotator_ranges[x][0] for x in annotator_set - {annotator}])
            others_right = max([annotator_ranges[x][1] for x in annotator_set - {annotator}])
            overlap = get_overlap_proportion(left, right, others_left, others_right)
            lowest_overlap = min(lowest_overlap, overlap)
            if overlap < MORE_MIN_OVERLAP:
                wrong_annotators.add(annotator)
        # Special case for 2 annotators
        if (nbr_annotators == 2) and (lowest_overlap < TWOFER_MIN_OVERLAP):
            wrong_annotators.update(annotator_ranges.keys())

    rejected_annotators = len(wrong_annotators)
    rejected_all = rejected_annotators == nbr_annotators
    logging.warning(f'Identified {rejected_annotators} wrong-column annotators out of {nbr_annotators}')
    # mark all the matching lines
    muster_sheet['wrong_column'] = muster_sheet['annotator'].apply(lambda x: x in wrong_annotators)
    # https://stackoverflow.com/questions/53550988/count-occurrences-of-false-or-true-in-a-column-in-pandas
    # https://stackoverflow.com/questions/23403352/return-default-if-pandas-dataframe-loc-location-doesnt-exist
    wrong_column_rows = muster_sheet['wrong_column'].value_counts().get(True, 0)
    logging.warning(f'Marked {wrong_column_rows} rows as wrong-column')
    return muster_sheet, rejected_all


def basic_cleaning(dataframe: pd.DataFrame):
    """
    Delete the rows with missing rectangles, cleaned text, or required identifiers
    :param dataframe:
    :return:
    """
    logging.warning(f'Starting rows {dataframe.shape[0]}')
    # Note: the text_clean column contains a single space where the exported text is blank
    # but is empty (NaN here) where it ended up empty after stripping tags.
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['left', 'right', 'top', 'bottom', 'text_clean',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    clean_rows = dataframe.shape[0]
    logging.warning(f'Dropped NaN rectangles/text/IDs; filtered rows {clean_rows}')
    start_time = time.time()
    return dataframe, clean_rows, start_time


def make_levenshtein_series(text_series: pd.Series):
    """
    Generate a column of Levenshtein similarity ratios
    :param text_series: column of strings
    :return: column giving the similarity between each string and the previous one
    """
    texts = text_series.values
    output = [np.NAN] + [Levenshtein.ratio(texts[i - 1], texts[i]) for i in range(1, texts.shape[0])]
    return pd.Series(output, index=text_series.index)


def make_line_series(overlap_series: pd.Series, annotator_series: pd.Series) -> pd.Series:
    """
    Generate line numbers for one muster sheet (subject-workflow combination)
    :param annotator_series: column of annotator IDs
    :param overlap_series: column of vertical overlaps
    :return: column of line numbers
    """
    line = 0
    results = [line]
    current_annotators = {annotator_series.values[0]}
    # value 0 is unused
    for this_overlap, this_annotator in zip(overlap_series.values[1:], annotator_series[1:]):
        if (this_overlap < VERTICAL_OVERLAP_THRESHOLD) or (this_annotator in current_annotators):
            line += 1
            current_annotators = {this_annotator}
        else:
            current_annotators.add(this_annotator)
        results.append(line)
    return pd.Series(results, index=overlap_series.index, dtype=pd.Int64Dtype)


def add_line_numbers(muster_sheet: pd.DataFrame) -> pd.DataFrame:
    # discard wrong-column rows before numbering
    muster_sheet = muster_sheet[~ muster_sheet['wrong_column']]

    # This way of sorting produces a slightly cleaner separation in the histogram
    # for some examples (empirically).
    muster_sheet = muster_sheet.sort_values(by=['top', 'bottom'], ascending=[True, False])

    muster_sheet['prev_top'] = muster_sheet['top'].rolling(window=2).min()
    muster_sheet['prev_bottom'] = muster_sheet['bottom'].rolling(window=2).min()
    muster_sheet['levenshtein'] = make_levenshtein_series(muster_sheet['text_clean'])
    muster_sheet['v_overlap'] = muster_sheet.apply(
        lambda row: get_overlap_proportion(row['top'], row['bottom'], row['prev_top'], row['prev_bottom']), axis=1)
    muster_sheet['line'] = make_line_series(muster_sheet['v_overlap'], muster_sheet['annotator'])
    logging.warning(f'Lines 0 to {muster_sheet["line"].max()}')
    return muster_sheet


def process_big_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse the input file, number the lines, carry out voting
    :param dataframe: the big CSV unpacked and repacked from Zooniverse exports
    :return: same thing with more columns
    """
    dataframe, total_rows, start_time = basic_cleaning(dataframe)
    # TODO detect column jumping
    # example: subject 79888343, workflow 21955, annotator greeneagle245
    # Probably in the wrong-column detection code, using h_overlap within each annotator?

    # Process each muster sheet (subject-workflow combination) separately
    muster_grouping = dataframe.groupby(by=['subject_id', 'workflow_id'])

    # decision type, number of votes -> nbr of matching decisions
    decision_counter = defaultdict(Counter)

    total_sheets = len(muster_grouping.groups.keys())
    sheets_done = 0
    rows_done = 0
    for muster_key in muster_grouping.groups.keys():
        muster_sheet = muster_grouping.get_group(muster_key)
        sheet_rows = muster_sheet.shape[0]
        logging.info(f'Processing {muster_key} with {sheet_rows} rows')

        muster_sheet, rejected = detect_wrong_columns(muster_sheet)
        dataframe = dataframe.combine_first(muster_sheet[['wrong_column']])

        # Skip line-numbering and voting if the whole sheet is rejected (two annotators
        # with different columns).
        if not rejected:
            # Add line numbers (ignoring wrong_column annotators)
            muster_sheet = add_line_numbers(muster_sheet)
            # Vote for agreement
            muster_sheet = vote_line_by_line(muster_key, muster_sheet, decision_counter)
            # Now copy the changes back to the main dataframe
            dataframe = dataframe.combine_first(muster_sheet[['prev_top', 'prev_bottom', 'v_overlap',
                                                              'levenshtein', 'line', 'decision_type',
                                                              'decision', 'decision1', 'text_median',
                                                              'mode_votes', 'total_votes']])
        sheets_done += 1
        rows_done += sheet_rows
        update_progress(rows_done, total_rows, sheets_done, total_sheets, start_time)

    log_decision_counts(decision_counter)
    return dataframe


def update_progress(rows_done: int, total_rows:int ,
                    sheets_done:int, total_sheets: int, start_time:float):
    sheet_ratio = sheets_done / total_sheets
    row_ratio = rows_done / total_rows
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_by_rows = str(timedelta(seconds=round(elapsed_time / row_ratio + start_time - current_time)))
    remaining_by_sheets = str(timedelta(seconds=round(elapsed_time / sheet_ratio + start_time - current_time)))
    logging.warning(f'Approx progress and time left: {100 * row_ratio:.1f}% {remaining_by_rows} by rows; '
                    f'{100 * sheet_ratio:.1f}% {remaining_by_sheets} by sheets')
    return


def log_decision_counts(decision_counter: dict):
    for vote_type, stuff in decision_counter.items():
        for vote_size, nbr in stuff.most_common():
            logging.warning(f'Decision type {vote_type} from {vote_size} = {nbr}')
    return


def vote_line_by_line(muster_key: tuple, muster_sheet: pd.DataFrame,
                      decision_counter: dict) -> pd.DataFrame:
    line_grouping = muster_sheet.groupby(by=['line'])
    for line in line_grouping.groups.keys():
        muster_line = line_grouping.get_group(line)
        key_triple = (muster_key[0], muster_key[1], line)
        muster_line = decide_line(muster_line, key_triple, decision_counter)
        # Copy the new columns back to the output
        muster_sheet = muster_sheet.combine_first(
            muster_line[['decision', 'decision1', 'text_median', 'decision_type', 'mode_votes', 'total_votes']])
    return muster_sheet


def decide_line(muster_line: pd.DataFrame, key_triple: tuple, decision_counter: dict) -> pd.DataFrame:
    """
    Carry out the voting on one subject/workflow/line combination
    :param key_triple: for logging
    :param muster_line: one subject-workflow-line combination
    :param decision_counter: to keep track of decisions by type and size
    :return:
    """
    # This comes out automatically in descending order
    vote_df = muster_line['text_clean'].value_counts()
    mode = vote_df.index[0]
    mode_votes = vote_df.values[0]
    total_votes = vote_df.values.sum()
    # weighting may speed the median calculation up
    median = Levenshtein.median(vote_df.index, wlist=vote_df.values)
    muster_line['text_median'] = median

    if mode_votes == total_votes:
        decision_type = 'unanimous'
    elif mode_votes > total_votes / 2:
        decision_type = 'majority'
    elif mode_votes > vote_df.values[1]:
        decision_type = 'plurality'
    else:
        decision_type = 'failed'

    # log and apply the result
    if decision_type != 'unanimous':
        logging.info(f'Voting {decision_type}: {key_triple} has {vote_df.to_dict()} (median={median})')
    decision_counter[decision_type][total_votes] += 1
    if decision_type == 'failed':
        muster_line['decision'] = None
        muster_line['decision1'] = median
    else:
        muster_line['decision'] = mode
        muster_line['decision1'] = mode
    muster_line['decision_type'] = decision_type
    muster_line['mode_votes'] = mode_votes
    muster_line['total_votes'] = total_votes
    return muster_line


def get_absolute_overlap(left0, right0, left1, right1):
    """
    Calculate the overlap in pixels across two ranges
    :param left0: left or top
    :param right0: right or bottom
    :param left1: other left or top
    :param right1: other right or bottom
    :return: overlap in pixels
    """
    return max(0, min(right0, right1) - max(left0, left1))


def get_overlap_proportion(this_left, this_right, reference_left, reference_right):
    """
    Calculate the proportion of "this" width covered by the "reference"
    :param this_left: left or top
    :param this_right: right or bottom
    :param reference_left: other left or top
    :param reference_right: other right or bottom
    :return: overlap as a proportion
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
    logging.warning('Sorting results...')
    big_df.sort_values(by=['workflow_id', 'subject_id', 'line', 'top'],
                       ascending=[True, True, True, True],
                       na_position='last',
                       inplace=True)

    logging.warning(f'Writing {output_filename} ...')
    big_df.to_csv(output_filename)
    logging.warning('Finished!')

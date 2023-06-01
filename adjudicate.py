#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import re
import time
from collections import defaultdict, Counter
from datetime import timedelta
from typing import Dict

import Levenshtein
import numpy as np
import pandas as pd

# Minimum size for a box to be usable
PIXEL_CUTOFF = 5.0

# Line separation (determined empirically from the histogram)
VERTICAL_OVERLAP_THRESHOLD = 0.6

# Wrong column detection: minimum overlap to keep both of two annotators
TWOFER_MIN_OVERLAP = 0.7
# Wrong column detection: minimum overlap for one of more than two annotators
MORE_MIN_OVERLAP = 0.1

# used to extract vessel name and page sequence from JPG filename
FILENAME_PATTERN = re.compile(r'\d+_(\S+)_(\d+)\.\w+')

# Patterns for cleaning manual annotations
TAG_PURGE = re.compile(r'[\[{][^[{]*[}\]]')
TAG_MATCH = re.compile(r'[\[{\]}]')
SMART_SINGLE_QUOTE = re.compile('[\u2018\u2019]')
SMART_DOUBLE_QUOTE = re.compile('[\u201c\u201d]')
NA_PATTERN = re.compile(r'\s*N/|\s*', flags=re.IGNORECASE)

WHITESPACE = re.compile(r'\s+')

# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
# These types are all nullable
TYPE_SPEC = {'text': 'string', 'annotator': 'string', 'filename': 'string',
             'finished_str': 'string', 'task': 'string',
             'top': 'Int64', 'bottom': 'Int64', 'left': 'Int64', 'right': 'Int64',
             'workflow_id': 'Int64', 'subject_id': 'Int64',
             'classification_id': 'Int64', 'vessel_page': 'Int64'}

# columns to fill in with DataFrame.update from groupings
GENERATED_COLUMNS = ['decision', 'decision1', 'decision_type', 'levenshtein',
                     'line', 'mode_votes', 'prev_bottom', 'prev_top', 'text_median',
                     'total_annotations', 'total_votes', 'v_overlap', 'wrong_column']


def is_too_small(row: pd.Series) -> bool:
    """
    Is the annotated image box unreasonably small?
    :param row:
    :return:
    """
    return (row['width'] < PIXEL_CUTOFF) or (row['height'] < PIXEL_CUTOFF)


def clean_median(s0: str) -> str:
    """
    Strip/clean whitespace for median strings but fall back to a single space instead
    of an empty string
    :param s0:
    :return:
    """
    s1 = WHITESPACE.sub(' ', s0.strip())
    return s1 or ' '


def get_vessel_and_page(row: pd.Series) -> pd.Series:
    """
    Extract the vessel name and page sequence from the image filename
    :param row: dataframe row including 'filename' column
    :return: vessel, vessel_page
    """
    filename = row['filename']
    match = FILENAME_PATTERN.fullmatch(filename)
    if match:
        result = [match.group(1), int(match.group(2))]
    else:
        result = [None, None]
    return pd.Series(data=result, index=['vessel', 'vessel_page'])


def clean_text(s0: str) -> str:
    """
    Clean manual annotations (before voting and generating medians)
    :param s0: manual annotation
    :return: cleaned form
    """
    if pd.isnull(s0):
        return s0
    # Genuine empty strings need to be changed to one space to prevent automatic
    # conversion to NaN through CSV interchange and to allow HTR to work.
    if not s0:
        return ' '
    # literal "N/A", which pandas turns into NaN
    if NA_PATTERN.fullmatch(s0):
        return ' '
    # trim whitespace and check again
    s1 = WHITESPACE.sub(' ', s0.strip())
    s1 = s1 or ' '
    # Ignore tags and let the regex filters do the work!
    # if TAG_MATCH.search(s1):
    #     s1 = clean_tags(s1)
    # # If tag removal leaves an empty string, leave it empty so it will be
    # # Nan and ignored later
    # replace smart quotes
    s1 = SMART_SINGLE_QUOTE.sub("'", s1)
    s1 = SMART_DOUBLE_QUOTE.sub('"', s1)
    return s1


def clean_tags(s0: str) -> str:
    return TAG_PURGE.sub('', s0)


def get_exclusions(config0) -> set:
    """
    Load the external exclusion file if present and return the known bad annotation sets
    :param config0:
    :return: set of triples (subject_id, workflow_id, annotator)
    """
    result = set()
    exclusion_file = config0.get('exclusions')
    if exclusion_file:
        logging.info(f'Loading exclusions from {exclusion_file}')
        with open(exclusion_file, 'r') as f0:
            for line in f0.readlines():
                line1 = line.strip()
                if line1 and not line1.startswith('#'):
                    triple = line1.split(',', maxsplit=2)
                    if len(triple) < 3:
                        logging.warning(f'Bad line: {line1}')
                    else:
                        # workflow and subject IDs are numeric
                        triple1 = (int(triple[0]), int(triple[1]), triple[2])
                        logging.info(f'Exclusion: {triple1}')
                        result.add(triple1)
        logging.info(f'Nbr of exclusions: {len(result)}')
    else:
        logging.info('No exclusion file specified')
    return result


def detect_wrong_columns(muster_sheet: pd.DataFrame):
    """
    Detect annotators working in the wrong column
    :param muster_sheet: one subject-workflow combination
    :return: same with extra columns; boolean whether all annotators have been rejected
    """
    wrong_annotators = set()
    nbr_annotators = muster_sheet['annotator'].unique().size

    # You can't calculate overlaps with 1 annotator.
    # If there are two columns with poor overlap, we assume one of them is wrong
    # but can't decide which so we mark them both for exclusion.
    if nbr_annotators > 1:
        # key annotator -> tuple (min_left, max_right)
        annotator_ranges = dict()
        annotator_grouped = muster_sheet.groupby('annotator')
        for annotator in annotator_grouped.groups.keys():
            anno_df = annotator_grouped.get_group(annotator)
            annotator_ranges[annotator] = (anno_df['left'].min(), anno_df['right'].max())
        annotator_set = set(annotator_ranges.keys())
        lowest_overlap = 1.0
        # TODO maybe handle edge case with more than three annotators, two of which are wrong
        for annotator, (left, right) in annotator_ranges.items():
            other_annotators = annotator_set - {annotator}
            others_left = min([annotator_ranges[x][0] for x in other_annotators])
            others_right = max([annotator_ranges[x][1] for x in other_annotators])
            overlap = get_relative_proportion(left, right, others_left, others_right)
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


def basic_cleaning(dataframe: pd.DataFrame, exclusions0: set, filters0: dict):
    """
    Delete the rows with missing rectangles, cleaned text, or required identifiers
    :param filters0:
    :param exclusions0:
    :param dataframe:
    :return:
    """
    start_time = time.time()
    logging.warning(f'Starting rows {dataframe.shape[0]}')
    logging.info(f'Starting columns {";".join(dataframe.columns)}')
    # TODO eventually: use the muster date tasks instead of dropping them
    # We Now clean text here (rather than in convert_classifications)
    # Note: the text_clean column contains a single space where the exported text is blank
    # (required for SimpleHTR and to stop pandas NaN)
    dataframe.loc[:, 'text_clean'] = dataframe['text'].apply(clean_text)
    # drop inherently unusable rows
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['left', 'right', 'top', 'bottom', 'text_clean',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    nbr_clean_rows = dataframe.shape[0]
    logging.warning(f'Dropped NaN rectangles/text/IDs; filtered rows {nbr_clean_rows}')
    # add vessel names and page numbers extracted from image filenames
    dataframe.loc[:, ['vessel', 'vessel_page']] = dataframe.apply(get_vessel_and_page, axis=1)
    # apply regex filters according to workflow
    dataframe.loc[:, 'regex_ok'] = dataframe.apply(lambda row: regex_filter_row(row, filters0), axis=1)
    # flag manual exclusions and small boxes
    dataframe.loc[:, 'exclude'] = dataframe.apply(lambda row: exclude_manually(row, exclusions0), axis=1)
    dataframe.loc[:, 'too_small'] = dataframe.apply(is_too_small, axis=1)
    # convert timestamp in case we need it later
    dataframe.loc[:, 'timestamp'] = pd.to_datetime(dataframe['finished_str'], errors='coerce', utc=True)
    logging.info(f'Preprocessed columns {";".join(dataframe.columns)}')

    # add empty columns to be filled in with update
    dataframe = pd.concat([dataframe, pd.DataFrame(columns=GENERATED_COLUMNS)])
    return dataframe, nbr_clean_rows, start_time


def make_levenshtein_similarity_series(text_series: pd.Series):
    """
    Generate a column of Levenshtein similarity ratios
    :param text_series: column of strings
    :return: column giving the similarity between each string and the previous one
    """
    texts = text_series.values
    output = [np.NAN] + [Levenshtein.ratio(texts[i - 1], texts[i]) for i in range(1, texts.shape[0])]
    return pd.Series(output, index=text_series.index)


def make_line_number_series(overlap_series: pd.Series, annotator_series: pd.Series) -> pd.Series:
    """
    Generate line numbers for one muster sheet (subject-workflow combination, 1 or more annotators)
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


def add_line_numbers(muster_sheet: pd.DataFrame):
    """
    Add line numbers to one subject-workflow combination
    :param muster_sheet:
    :return:
    """
    # discard wrong-column rows and manually excluded annotation sets before numbering
    muster_sheet = muster_sheet[~ (muster_sheet['wrong_column'] | muster_sheet['exclude'])]

    # ignore sheets that have 0 valid annotations
    if muster_sheet.shape[0]:
        # This way of sorting produces a slightly cleaner separation in the histogram
        # for some examples (empirically).
        muster_sheet = muster_sheet.sort_values(by=['top', 'bottom'], ascending=[True, False])

        muster_sheet['prev_top'] = muster_sheet['top'].shift()
        muster_sheet['prev_bottom'] = muster_sheet['bottom'].shift()
        muster_sheet['levenshtein'] = make_levenshtein_similarity_series(muster_sheet['text_clean'])
        muster_sheet['v_overlap'] = muster_sheet.apply(
            lambda row: get_relative_proportion(row['top'], row['bottom'], row['prev_top'], row['prev_bottom']), axis=1)
        muster_sheet['line'] = make_line_number_series(muster_sheet['v_overlap'], muster_sheet['annotator'])
        logging.warning(f'Lines 0 to {muster_sheet["line"].max()}')
        rejected = False
    else:
        logging.warning(f'Muster sheet empty after cleaning')
        rejected = True
    return muster_sheet, rejected


def exclude_manually(row: pd.Series, exclusions0: set) -> bool:
    """
    Is this subject-workflow-annotator combination in the manual exclusion list?
    :param row:
    :param exclusions0:
    :return:
    """
    key = (row['subject_id'], row['workflow_id'], row['annotator'])
    return key in exclusions0


def process_big_df(dataframe: pd.DataFrame, exclusions0, filters0) -> pd.DataFrame:
    """
    Analyse the input file; split into subject-workflow combinations; number the lines and carry out voting;
    copy everything back onto the dataframe.
    :param filters0: dict workflow int -> compiled regex
    :param exclusions0:
    :param dataframe: the big CSV unpacked and repacked from Zooniverse exports
    :return: same thing with more columns
    """
    dataframe, total_rows, start_time = basic_cleaning(dataframe, exclusions0, filters0)
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

        muster_sheet, all_rejected = detect_wrong_columns(muster_sheet)

        # Skip line-numbering and voting if the whole sheet is rejected (two annotators
        # with different columns).
        if not all_rejected:
            # Add line numbers (ignoring wrong_column annotators)
            muster_sheet, all_excluded = add_line_numbers(muster_sheet)
            if not all_excluded:
                # Vote for agreement
                muster_sheet = vote_line_by_line(muster_key, muster_sheet, decision_counter)
                # Now copy the changes back to the main dataframe
                dataframe.update(muster_sheet[['prev_top', 'prev_bottom', 'v_overlap',
                                               'levenshtein', 'line', 'decision_type',
                                               'decision', 'decision1', 'text_median',
                                               'total_annotations', 'mode_votes',
                                               'total_votes', 'wrong_column']])
            else:
                dataframe.update(muster_sheet[['wrong_column']])
        else:
            dataframe.update(muster_sheet[['wrong_column']])
        sheets_done += 1
        rows_done += sheet_rows
        update_progress(rows_done, total_rows, sheets_done, total_sheets, start_time)

    log_decision_counts(decision_counter)
    return dataframe


def regex_filter_row(row: pd.Series, filters0: Dict[int, re.Pattern]) -> bool:
    """
    Does this cleaned text pass the regex filter for the workflow?
    :param row:
    :param filters0:
    :return:
    """
    text = row['text_clean']
    workflow_id = row['workflow_id']
    pattern = filters0.get(workflow_id)
    if pattern:
        return bool(pattern.fullmatch(text))
    # pass if there is no filter for the given workflow
    return True


def update_progress(rows_done: int, total_rows: int,
                    sheets_done: int, total_sheets: int, start_time: float):
    """
    Log a rough indication of progress and time left to finish.
    :param rows_done:
    :param total_rows:
    :param sheets_done:
    :param total_sheets:
    :param start_time:
    :return:
    """
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
    """
    Log a summary of the voting results over the whole thing.
    :param decision_counter:
    :return:
    """
    for vote_type, stuff in decision_counter.items():
        for vote_size, nbr in stuff.most_common():
            logging.warning(f'Decision type {vote_type} from {vote_size} = {nbr}')
    return


def vote_line_by_line(muster_key: tuple, muster_sheet: pd.DataFrame,
                      decision_counter: dict) -> pd.DataFrame:
    """
    Carry out the voting for a subject-workflow combination.
    :param muster_key:
    :param muster_sheet:
    :param decision_counter:
    :return:
    """
    line_grouping = muster_sheet.groupby(by=['line'])
    for line in line_grouping.groups.keys():
        muster_line = line_grouping.get_group(line)
        key_triple = (muster_key[0], muster_key[1], line)
        muster_line = vote_one_line(muster_line, key_triple, decision_counter)
        # Copy the new columns back to the output
        muster_sheet.update(muster_line[['decision', 'decision1', 'text_median', 'decision_type',
                                         'mode_votes', 'total_votes', 'total_annotations']])
    return muster_sheet


def vote_one_line(muster_line: pd.DataFrame, key_triple: tuple, decision_counter: dict) -> pd.DataFrame:
    """
    Carry out the voting on one subject-workflow-line combination
    :param key_triple: for logging
    :param muster_line: one subject-workflow-line combination
    :param decision_counter: to keep track of decisions by type and size
    :return:
    """
    # Do not count regex_ok==False annotations as votes --- but they can be used
    # for JPG boxes if there are regex_ok==True annotations for the same line of the muster sheet
    total_annotations = muster_line.shape[0]
    muster_line['total_annotations'] = total_annotations
    votable = muster_line[muster_line['regex_ok']]

    if votable.empty:
        muster_line['decision'] = None
        muster_line['decision1'] = None
        muster_line['decision_type'] = 'empty'
        muster_line['mode_votes'] = 0
        muster_line['total_votes'] = 0
        muster_line['text_median'] = None
        logging.info(f'Voting empty: {key_triple}')
    else:
        # This comes out automatically in descending order
        vote_df = votable['text_clean'].value_counts()
        mode = vote_df.index[0]
        mode_votes = vote_df.values[0]
        total_votes = vote_df.values.sum()
        # weighting may speed the median calculation up
        median = clean_median(Levenshtein.median(vote_df.index, wlist=vote_df.values))
        muster_line['text_median'] = median

        if mode_votes == total_annotations:
            decision_type = 'unanimous'
        elif mode_votes == total_votes:
            decision_type = 'pseudo-unanimous'
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
    # TODO make this more efficient by returning pd.Series and using df.apply(...) in the calling function
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


def get_relative_proportion(this_left, this_right, reference_left, reference_right):
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

    oparser.add_argument('-f', dest='filter_files',
                         default=[],
                         action='append',
                         metavar='FILTER.json',
                         help='filter file (can be repeated)')

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

    filters = {}

    for filter_file in options.filter_files:
        with open(filter_file, 'r') as f:
            addenda = json.load(f)
            # force keys to be int for compatibility later
            filters.update({int(k): v for k, v in addenda.items()})

    for k, v in filters.items():
        logging.info(f'Filter: {k} {v}')

    compiled_filters = {k: re.compile(v) for k, v in filters.items()}

    input_filename = config['intermediate_file']
    output_filename = config['agreed_file']
    exclusions = get_exclusions(config)

    logging.info(f'Loading {input_filename}')
    big_df = pd.read_csv(input_filename, dtype=TYPE_SPEC)
    big_df = process_big_df(big_df, exclusions, compiled_filters)

    # sort the output to make it easy to inspect the voting and overlaps
    logging.warning('Sorting results...')
    big_df.sort_values(by=['workflow_id', 'subject_id', 'line', 'top'],
                       ascending=[True, True, True, True],
                       na_position='last',
                       inplace=True)

    logging.warning(f'Final rows {big_df.shape[0]}')
    logging.warning(f'Final columns {";".join(big_df.columns)}')
    logging.warning(f'Writing {output_filename} ...')
    big_df.to_csv(output_filename)
    logging.warning('Finished!')

#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import re
from collections import defaultdict, Counter

import pandas as pd


WHITESPACE = re.compile(r'\s+')


def clean_string(s0: str):
    s1 = s0.strip()
    return WHITESPACE.sub(' ', s1)


def process_big_df(dataframe):
    """
    Analyse the input file, generate line numbers, and produce a quasi-database of the annotations
    :param dataframe:
    :return: nested dict  tuple(subject, workflow, line) -> annotator -> tuple(annotation, x, width)
    """
    logging.warning(f'Starting rows {dataframe.shape[0]}')
    # delete the rows without rectangles, text, or identifiers
    dataframe.dropna(axis=0, how='any', inplace=True, subset=['x', 'y', 'width', 'height', 'text',
                                                              'subject_id', 'workflow_id', 'annotator', 'task'])
    logging.warning(f'Dropped NaN rectangles/text/IDs; filtered rows {dataframe.shape[0]}')

    # clean the ends of the strings and suppress newlines
    dataframe['text_clean'] = dataframe['text'].apply(clean_string)
    logging.warning(f'Cleaned text strings')
    # create column with special value to start it as int64
    dataframe['line'] = -1

    # put all the annotations into a quasi-database
    # key: tuple(subject, workflow, line)
    # value: dict annotator -> (annotation, x, width)
    annotation_store0 = defaultdict(dict)

    # split into subunits by subject and workflow IDs and annotator
    grouped_columns = dataframe.groupby(by=['subject_id', 'workflow_id', 'annotator'])
    for muster_column in grouped_columns.groups.keys():
        muster_column_raw = grouped_columns.get_group(muster_column)
        logging.info(f'Processing {muster_column} with {muster_column_raw.shape[0]} rows')
        # create a new DF so we can reset the index and add a column
        muster_column_df = muster_column_raw.sort_values(by='y', ascending=True)
        # index must be reset for the pd range sequence to work
        muster_column_df.reset_index(inplace=True, drop=True)
        muster_column_df['line'] = pd.Series(range(muster_column_df.shape[0]))
        for i, row in muster_column_df.iterrows():
            subject_id = row['subject_id']
            workflow_id = row['workflow_id']
            line = row['line']
            y = row['y']
            # add this manual annotation to the annotation store
            key = (subject_id, workflow_id, line)
            annotator = row['annotator']
            annotation = row['text_clean']
            annotation_store0[key][annotator] = (annotation, row['x'], row['width'])
            # add the line number back to the main dataframe for use later
            main_index = dataframe.index[(dataframe['subject_id'] == subject_id) &
                                         (dataframe['workflow_id'] == workflow_id) &
                                         (dataframe['annotator'] == annotator) &
                                         (dataframe['y'] == y)
                                         ].tolist()[0]
            dataframe.at[main_index, 'line'] = line
    logging.warning(f'Total number of annotated muster lines {len(annotation_store0)}')
    return annotation_store0


def process_database(annotation_store0: dict):
    """
    Carry out the voting on each subject-workflow-line combination
    :param annotation_store0: tuple(subject, workflow, line) -> dict annotator -> (annotation, x, width)
    :return: tuple(subject, workflow, line) -> tuple(annotation or None, mode votes, total votes, anno_overlaps)
    """
    # decisions value: tuple(annotation or None, mode votes, total votes, annotator_overlaps)
    decisions0 = dict()
    # decision type, number of votes -> nbr of matching decisions
    decision_counter = defaultdict(Counter)
    for key, annotation_dict in annotation_store0.items():
        voting_results = decide(key, annotation_dict, decision_counter)
        decisions0[key] = voting_results
        # decision = None means no majority so no result
    total = 0
    for vote_type, stuff in decision_counter.items():
        subtotal = 0
        for vote_size, nbr in stuff.most_common():
            subtotal += nbr
            logging.warning(f'Decision type {vote_type} from {vote_size} = {nbr}')
        logging.warning(f'Decision type {vote_type} subtotal = {subtotal}')
        total += subtotal
    logging.warning(f'Decision total = {total}')
    return decisions0


def decide(key_triple: tuple, annotation_dict: dict, decision_counter: dict):
    """
    Carry out the voting on one subject/workflow/line combination
    :param key_triple: subject, workflow, line
    :param annotation_dict: annotator -> (annotation, x, width)
    :param decision_counter: to keep track of decisions by type and size
    :return:
    """
    annotator_overlap_ratings = get_group_overlaps(annotation_dict)
    # Ignore "wrong column" annotations BUT only if practical;
    # i.e., do not ignore everything. If there are two disjoint
    # rectangles, they will both have zero overlap with each other.
    ignore = {annotator for annotator, overlap in annotator_overlap_ratings.items() if overlap == 0.0}
    if len(ignore) == len(annotation_dict):
        ignore = set()

    c = Counter([text[0] for annotator, text in annotation_dict.items() if annotator not in ignore])
    # indicate ignored in output

    mc = c.most_common()
    mode, mode_votes = mc[0]
    total_votes = len(annotation_dict)
    if mode_votes == total_votes:
        decision_counter['unanimous'][total_votes] += 1
        result = mode
    elif mode_votes > total_votes / 2:
        decision_counter['majority'][total_votes] += 1
        logging.info(f'Voting majority: {key_triple} has {c.most_common()}')
        log_annotation_disagreement(annotation_dict, annotator_overlap_ratings)
        result = mode
    elif mode_votes > mc[1][1]:
        decision_counter['plurality'][total_votes] += 1
        logging.info(f'Voting plurality: {key_triple} has {c.most_common()}')
        log_annotation_disagreement(annotation_dict, annotator_overlap_ratings)
        result = mode
    else:  # dodgy
        decision_counter['failed'][total_votes] += 1
        logging.info(f'Voting failed: {key_triple} has {c.most_common()}')
        log_annotation_disagreement(annotation_dict, annotator_overlap_ratings)
        result = None
    return result, mode_votes, total_votes, annotator_overlap_ratings


def log_annotation_disagreement(annotation_dict: dict, annotator_overlap_ratings: dict) -> None:
    """
    Log/print a simplified dict showing the votes and overlap score for each annotator
    :param annotation_dict:
    :param annotator_overlap_ratings:
    :return:
    """
    result = {annotator: (value[0], annotator_overlap_ratings[annotator])
              for annotator, value in annotation_dict.items()}
    logging.info(f'  from {result}')
    return


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


def get_group_overlaps(annotation_dict: dict) -> dict:
    """
    Calculate the relative overlap for each annotator with respect to
    groups of the others
    :param annotation_dict: annotator -> (annotation, x, width)
    :return: annotator -> overlap compared with all others
    """
    if len(annotation_dict) == 1:
        return {list(annotation_dict.keys())[0]: 1.0}
    # Note: one vs others does not work for a singleton!
    annotator_boundaries = get_horizontal_boundaries(annotation_dict)
    # annotator -> (left, right)
    all_annotators = annotator_boundaries.keys()
    output = dict()
    for annotator, boundaries in annotator_boundaries.items():
        others = all_annotators - {annotator}
        others_left = min([annotator_boundaries.get(x)[0] for x in others])
        others_right = max([annotator_boundaries.get(x)[1] for x in others])
        output[annotator] = get_overlap_proportion(boundaries[0], boundaries[1], others_left, others_right)
    return output


def apply_decisions(dataframe, decisions0) -> pd.DataFrame:
    dataframe = dataframe.join(dataframe.apply(apply_one_decision, axis=1, decision_dict=decisions0))
    logging.warning(f"filled in decision = {dataframe['decision'].count()}")
    logging.warning(f"filled in line = {dataframe['line'].count()}")
    logging.warning(f'Dataframe shape = {dataframe.shape}')
    # apply makes these float64 for some reason
    dataframe['mode_votes'] = dataframe['mode_votes'].astype('int64')
    dataframe['total_votes'] = dataframe['total_votes'].astype('int64')
    return dataframe


def apply_one_decision(row: pd.Series, decision_dict: dict) -> pd.Series:
    """
    Look up the voting results for this task and get the voting results and
    related information
    :param row:
    :param decision_dict:
    :return: horizontal series to be added to the main DF row
    """
    key = (row['subject_id'], row['workflow_id'], row['line'])
    result, mode_votes, total_votes, annotator_overlaps = decision_dict.get(key, (None, 0, 0, dict()))
    annotator = row['annotator']
    overlap = annotator_overlaps.get(annotator, 0.0)
    return pd.Series([result, mode_votes, total_votes, overlap],
                     index=['decision', 'mode_votes', 'total_votes', 'overlap'])


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

    annotation_store = process_big_df(big_df)
    decisions = process_database(annotation_store)

    big_df1 = apply_decisions(big_df, decisions)
    logging.warning(f'Writing {output_filename}')

    # sort the output to make it easy to inspect the voting and overlaps
    big_df1.sort_values(by=['workflow_id', 'subject_id', 'line', 'overlap'],
                        ascending=[True, True, True, False],
                        na_position='last',
                        inplace=True)

    big_df1.to_csv(output_filename)

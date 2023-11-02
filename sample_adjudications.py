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

# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
# These types are all nullable
TYPE_SPEC = {'annotator': 'string', 'decision': 'string', 'decision1': 'string',
             'decision_type': 'string', 'filename': 'string',
             'text': 'string', 'text_clean': 'string', 'text_median': 'string',
             'vessel': 'string', 'finished_str': 'string', 'task': 'string',
             'top': 'Int64', 'bottom': 'Int64', 'left': 'Int64', 'right': 'Int64',
             'mode_votes': 'Int64', 'total_votes': 'Int64', 'total_annotations': 'Int64', 'line': 'Int64',
             'workflow_id': 'Int64', 'subject_id': 'Int64', 'classification_id': 'Int64',
             'exclude': 'boolean', 'regex_ok': 'boolean', 'too_small': 'boolean',
             'wrong_column': 'boolean'}
PARSE_DATES = ['timestamp']

oparser = argparse.ArgumentParser(description="draw a small sample of agreed annotations",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-o', dest='output_file',
                     type=str,
                     required=True,
                     metavar='FILE',
                     help='output file')

oparser.add_argument('-i', dest='input_file',
                     type=str,
                     required=True,
                     metavar='FILE',
                     help='input file')

oparser.add_argument('-n', dest='count',
                     type=int,
                     default=1000,
                     metavar='INT',
                     help='output size')

options = oparser.parse_args()

print(f'Reading {options.input_file}')
big_df = pd.read_csv(options.input_file, dtype=TYPE_SPEC,
                     parse_dates=PARSE_DATES, index_col=0)
print(f'Number of rows {big_df.shape[0]}')

sample = big_df.sample(n=options.count).sort_index()
print(f'Sampled rows: {sample.shape[0]}')

print(f'Writing {options.output_file}')
sample.to_csv(options.output_file)
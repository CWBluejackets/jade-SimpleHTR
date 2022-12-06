#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os

import pandas as pd

oparser = argparse.ArgumentParser(description="analyse the annotations processed from the exports",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str,
                     required=True,
                     metavar='FILE.json',
                     help='config file')

oparser.add_argument('-o', dest='output_file',
                     type=str,
                     required=True,
                     metavar='FILE.csv',
                     help='output file')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

# turn off the chained assignment warnings
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

log_directory = config.get('log_directory')

if log_directory:
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    logfile = os.path.join(config['log_directory'], f'{timestamp}_analyse.log')
    print(f'Logfile {logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:
    print(f'Log stdout')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

input_filename = config['intermediate_file']

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)
big_df['datetime'] = pd.to_datetime(big_df['finished_str'])
big_df['date'] = big_df['datetime'].dt.date

dateless_grouping = big_df.groupby(['annotator', 'workflow_id'])['classification_id'].agg(['count'])
dateless_grouping.reset_index(inplace=True)
dateless_pivot = dateless_grouping.pivot(index='annotator', columns='workflow_id',
                                         values='count').fillna(value=0).astype(int)

logging.info(f'Writing {options.output_file}')
dateless_pivot.to_csv(options.output_file)

anno_week_grouping = big_df.groupby(by=['annotator',
                                        pd.Grouper(key='datetime',
                                                   freq='W')])['classification_id'].agg(['count'])
anno_week_grouping.reset_index(inplace=True)
anno_week_grouping['date'] = anno_week_grouping['datetime'].dt.date
anno_week_pivot = anno_week_grouping.pivot(index='annotator', columns='date',
                                           values='count').fillna(0).astype(int)

aw_file_parts = os.path.splitext(os.path.basename(options.output_file))
aw_file = os.path.join(os.path.dirname(options.output_file),
                       f'{aw_file_parts[0]}_anno_weekly{aw_file_parts[1]}')

logging.info(f'Writing {aw_file}')
anno_week_pivot.to_csv(aw_file)

wf_week_grouping = big_df.groupby(by=['workflow_id',
                                      pd.Grouper(key='datetime',
                                                 freq='W')])['classification_id'].agg(['count'])
wf_week_grouping.reset_index(inplace=True)
wf_week_grouping['date'] = wf_week_grouping['datetime'].dt.date
wf_week_pivot = wf_week_grouping.pivot(index='workflow_id', columns='date',
                                       values='count').fillna(0).astype(int)

ww_file_parts = os.path.splitext(os.path.basename(options.output_file))
ww_file = os.path.join(os.path.dirname(options.output_file),
                       f'{ww_file_parts[0]}_wf_weekly{ww_file_parts[1]}')

logging.info(f'Writing {ww_file}')
wf_week_pivot.to_csv(ww_file)

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

oparser.add_argument('-i', dest='interval',
                     default=None,
                     metavar='INTERVAL',
                     help='optional column interval')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

# turn off the chained assignment warnings
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

log_directory = config.get('log_directory')

if log_directory:
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
    print(f'Logfile f{logfile}')
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
else:
    print(f'Log stdout')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

input_filename = config['intermediate_file']

logging.info(f'Loading {input_filename}')
big_df = pd.read_csv(input_filename)
big_df['date'] = pd.to_datetime(big_df['finished_str']).dt.date

dateless_grouping = big_df.groupby(['annotator', 'workflow_id'])['classification_id'].agg(['count'])
dateless_grouping.reset_index(inplace=True)
dateless_pivot = dateless_grouping.pivot(index='annotator', columns='workflow_id',
                                         values='count').fillna(value=0)

# TODO interval grouper
# columns = [min, meanr, medianr, max]
# df = df.groupby(pd.Grouper(key='timestamp', freq=config1['averaging'])).agg({'temperature': columns})

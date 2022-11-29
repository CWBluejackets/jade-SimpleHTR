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
big_df['datetime'] = pd.to_datetime(big_df['finished_str'])
big_df['date'] = big_df['datetime'].dt.date

dateless_grouping = big_df.groupby(['annotator', 'workflow_id'])['classification_id'].agg(['count'])
dateless_grouping.reset_index(inplace=True)
dateless_pivot = dateless_grouping.pivot(index='annotator', columns='workflow_id',
                                         values='count').fillna(value=0)

logging.info(f'Writing {options.output_file}')

dateless_pivot.to_csv(options.output_file)

week_grouping = big_df.groupby(by=['annotator',
                                   pd.Grouper(key='dt',
                                              freq='W')])['classification_id'].agg(['count'])
week_grouping.reset_index(inplace=True)
week_pivot = week_grouping.pivot(index='annotator', columns='wk', values='count').fillna(0)

output_file_parts = os.path.splitext(os.path.basename(options.output_file))
week_output_file = os.path.join(os.path.dirname(options.output_file),
                                f'{output_file_parts[0]}_weekly{output_file_parts[1]}')

logging.info(f'Writing {week_output_file}')

week_pivot.to_csv(week_output_file)





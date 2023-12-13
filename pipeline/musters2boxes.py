#!/usr/bin/env python3

import argparse
from datetime import datetime
import itertools
import json
import logging
import os.path
import cv2
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
from collections import Counter

oparser = argparse.ArgumentParser(description="Process muster sheets for ML application",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

oparser.add_argument('-c', dest='config_file',
                     type=str, required=True,
                     metavar='config.json',
                     help='config file')

oparser.add_argument('-i', dest='input_location',
                     type=str, default=None,
                     metavar='FILE_OR_DIR',
                     help='input file or directory')

options = oparser.parse_args()

with open(options.config_file) as f:
    config = json.load(f)

    log_directory = config.get('log_directory')

    if log_directory:
        timestamp = datetime.now().isoformat(timespec='seconds')
        logfile = os.path.join(config['log_directory'], f'{timestamp}_create.log')
        print(f'Logfile {logfile}')
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        print(f'Log stdout')
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Use options.input_location if specified; otherwise fall back to config['input_directory'];
# options.input_location can be a single JPG or a directory containing *.jpg or *.JPG.

input_location = options.input_location or config['input_directory']
if os.path.isdir(input_location):
    # input_files = input_location/**.jpg JPG
    pass
elif os.path.isfile(input_location):
    input_files = [input_location]






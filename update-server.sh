#!/bin/bash

# directory where this file is located
SOURCE_DIR=$(dirname "$0")

TARGET_DIR="/jmain02/home/J2AD003/txk64/axf12-txk64/bluejackets/jade-SimpleHTR"

rsync -a -e ssh -v --delete  --delete-excluded \
      --exclude=.git  --exclude=.idea \
      "${SOURCE_DIR}/" "jade2:${TARGET_DIR}/"

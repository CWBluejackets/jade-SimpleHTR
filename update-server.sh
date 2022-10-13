#!/bin/bash

SOURCE_DIR=$(dirname "$0")

TARGET_DIR="/jmain02/home/J2AD003/txk64/axf12-txk64/bluejackets/jade-SimpleHTR"

rsync -a -e ssh -v --delete "${SOURCE_DIR}/" "jade2:${TARGET_DIR}/"

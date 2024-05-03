#!/usr/bin/env bash

conda activate simplehtr1

HTR_DIR="${HOME}/sandboxes/bluejackets-jade-SimpleHTR"

CONFIG_FILE="${HTR_DIR}/pipeline-config.json"
LOCATIONS_FILE="${HTR_DIR}/locations-laptop.json"
OPTIONS="-c  ${CONFIG_FILE}  -l ${LOCATIONS_FILE}  -v  -d  -r"

cd "${HTR_DIR}/src" || exit 99

COMMAND="python3 muster_pipeline.py ${OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}

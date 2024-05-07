#!/usr/bin/env bash

HTR_DIR="${HOME}/sandboxes/bluejackets-jade-SimpleHTR"

CONDA_ENV="simplehtr1"
CONFIG_FILE="${HTR_DIR}/pipeline-config.json"
LOCATIONS_FILE="${HTR_DIR}/locations-laptop.json"
OPTIONS="-c  ${CONFIG_FILE}  -l ${LOCATIONS_FILE}  -v  -d  -r"

cd "${HTR_DIR}/src" || exit 99

COMMAND="conda run  -n ${CONDA_ENV}  --live-stream  python3 muster_pipeline.py  ${OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}

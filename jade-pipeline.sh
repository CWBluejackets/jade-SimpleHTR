#!/usr/bin/env bash
# Request 5 gigabytes of real memory (mem)
#$ -l rmem=5G

# set memory, nodes, GPUs, time limit
#SBATCH --mem=16000
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# time limit on big is 24h <http://docs.jade.ac.uk/en/latest/jade/scheduler/>

#SBATCH --job-name=Muster_Pipeline
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.funk@sheffield.ac.uk

# INSTRUCTIONS
# http://docs.jade.ac.uk/en/latest/jade/scheduler/index.html

# load the modules for the program we want to run
module load python/anaconda3
module load cuda/10.1
echo "modules loaded"

HTR_DIR="${HOME}/bluejackets/jade-SimpleHTR"

# https://www.golinuxcloud.com/bash-getopts/
# https://www.computerhope.com/unix/bash/getopts.htm

# OPTIONS passed to SimpleHTR
# required
# -c convert fractions
# -d data control file
# -m model subdirectory
# -w wide_mode

CONFIG_FILE="$(realpath pipeline-config.json)}"
LOCATIONS_FILE="$(realpath locations-jade.json)}"
OPTIONS="-c  ${CONFIG_FILE}  -l ${LOCATIONS_FILE}"

while getopts "vdi:" option
do
  case "${option}" in
    v)
      # verbose (not actually used yet)
      OPTIONS="${OPTIONS} -v"
      ;;
    d)
      # debug: store cell images with output
      OPTIONS="${OPTIONS} -d"
      ;;
    i)
      echo "${OPTARG}"
      INPUT_LOCATION="$(realpath "${OPTARG}")"
      OPTIONS="${OPTIONS}  -i ${INPUT_LOCATION}"
      ;;
    *)
      echo "invalid option: ${option}"
      exit 13
  esac
done

source activate simplehtr1
echo "environment activated"
echo
cd "${HTR_DIR}/src" || exit 99

COMMAND="python3 muster_pipeline.py ${OPTIONS} --img_list_file ${DATA} --model_dir ${MODEL_DIR} --output_file ${OUTPUT_FILE}"
echo "${COMMAND}"
echo
${COMMAND}

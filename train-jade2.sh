#!/usr/bin/env bash
# Request 5 gigabytes of real memory (mem)
#$ -l rmem=5G

# set memory, nodes, GPUs, time limit
#SBATCH --mem=16000
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# time limit on big is 48h <http://docs.jade.ac.uk/en/latest/jade/scheduler/index.html#submitting-jobs-with-the-command-sbatch>

#SBATCH --job-name=SimpleHTR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.funk@sheffield.ac.uk

# INSTRUCTIONS
# http://docs.jade.ac.uk/en/latest/jade/scheduler/index.html

# load the modules for the program we want to run
module load python/anaconda3
module load cuda/10.1
echo "modules loaded"

HTR_DIR="${HOME}/bluejackets/jade-SimpleHTR"

source activate simplehtr1
echo "environment activated"
echo
cd "${HTR_DIR}/src" || exit 99

# https://www.golinuxcloud.com/bash-getopts/
# https://www.computerhope.com/unix/bash/getopts.htm

# OPTIONS passed to SimpleHTR
# -h -> --high_contrast
# -c -> --convert_fractions
# TODO -d DEBUG
# required
# -t 22014_1.00 = training subdirectory

MAIN_OPTIONS="--mode train --batch_size 500 --early_stopping 15"
MODEL_SUFFIX=""

while getopts "hct:" option
do
  case "${option}" in
    h)
      MAIN_OPTIONS="${MAIN_OPTIONS} --high_contrast"
      MODEL_SUFFIX="${MODEL_SUFFIX}_H"
      ;;
    c)
      MAIN_OPTIONS="${MAIN_OPTIONS} --convert_fractions"
      MODEL_SUFFIX="${MODEL_SUFFIX}_C"
      ;;
    t)
      TRAINING_SUBDIR="${OPTARG}"
      ;;
    *)
      echo "invalid option: ${option}"
      exit 13
  esac
done

TRAINING_DIR="${HOME}/bluejackets/training_data/${TRAINING_SUBDIR}}}"
MODEL_DIR="${HOME}/bluejackets/models/${TRAINING_SUBDIR}${MODEL_SUFFIX}"
DEBUG_DIR="${HOME}/bluejackets/debug/${TRAINING_SUBDIR}${MODEL_SUFFIX}"

echo "Training: ${TRAINING_DIR}"
echo "Model: ${MODEL_DIR}"
echo

# create if new; empty out if re-used
rm -fr "${MODEL_DIR}"
mkdir -vp "${MODEL_DIR}"
# TODO only if debug option is requested
# rm -fr "${DEBUG_DIR}"
# mkdir -vp "${DEBUG_DIR}"
echo

COMMAND="python3 main.py ${MAIN_OPTIONS} --data_dir ${TRAINING_DIR} --model_dir ${MODEL_DIR}"
echo "${COMMAND}"
"${COMMAND}"

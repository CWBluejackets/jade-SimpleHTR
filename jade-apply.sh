#!/usr/bin/env bash
# Request 5 gigabytes of real memory (mem)
#$ -l rmem=5G

# set memory, nodes, GPUs, time limit
#SBATCH --mem=16000
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# time limit on big is 24h <http://docs.jade.ac.uk/en/latest/jade/scheduler/>

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
# required
# -a path/to/words.txt data list
# -m model model subdirectory

OPTIONS="--mode apply "

while getopts "a:m:" option
do
  case "${option}" in
    a)
      LIST_FILE="${OPTARG}"
      ;;
    m)
      MODEL="${OPTARG}"
      ;;
    *)
      echo "invalid option: ${option}"
      exit 13
  esac
done

OUTPUT_FILE="${HOME}/bluejackets/test_${SLURM_JOB_ID}.csv"
MODEL_DIR="${HOME}/bluejackets/models/${MODEL}"

echo "Output file: ${OUTPUT_FILE}"
echo "Model: ${MODEL_DIR}"
echo "Word list: ${LIST_FILE}"
echo

COMMAND="python3 main.py ${OPTIONS} --img_list_file ${LIST_FILE} --model_dir ${MODEL_DIR} --output_file ${OUTPUT_FILE}"
echo "${COMMAND}"
echo
${COMMAND}

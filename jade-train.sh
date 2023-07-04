#!/usr/bin/env bash
# Request 5 gigabytes of real memory (mem)
#$ -l rmem=5G

# set memory, nodes, GPUs, time limit
#SBATCH --mem=16000
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# wall time limit on small is 6 days

#SBATCH --job-name=SimpleHTR_train
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

# https://www.golinuxcloud.com/bash-getopts/
# https://www.computerhope.com/unix/bash/getopts.htm

# OPTIONS passed to SimpleHTR_train
# -h -> --high_contrast
# -c -> --convert_fractions
# -a -> --augment
# -d -> --debug_dir DIR
# -w wide_mode
# -L --> --opt_lr VALUE
# -E --> --opt_e VALUE
# -D --> --dropout VALUE
# required
# -t 22014_1.00/words.txt = training control file

OPTIONS="--mode train --batch_size 500 --early_stopping 15"

while getopts "ahcwt:dL:E:D:" option
do
  case "${option}" in
    h)
      OPTIONS="${OPTIONS} --high_contrast"
      ;;
    c)
      OPTIONS="${OPTIONS} --convert_fractions"
      ;;
    w)
      OPTIONS="${OPTIONS} --wide_mode"
      ;;
    a)
      OPTIONS="${OPTIONS} --augment"
      ;;
    d)
      DEBUG_DIR="${HOME}/bluejackets/debug/${SLURM_JOB_ID}"
      OPTIONS="${OPTIONS} --debug_dir ${DEBUG_DIR}"
      mkdir -vp "${DEBUG_DIR}"
      echo "Debug: ${DEBUG_DIR}"
      ;;
    t)
      # shellcheck disable=SC2006
      TRAINING_FILE=$(realpath "${OPTARG}")
      ;;
    L)
      # learning rate
      OPTIONS="${OPTIONS} --opt_lr ${OPTARG}"
      ;;
    E)
      # epsilon
      OPTIONS="${OPTIONS} --opt_e ${OPTARG}"
      ;;
    D)
      # dropout
      OPTIONS="${OPTIONS} --dropout ${OPTARG}"
    *)
      echo "invalid option: ${option}"
      exit 13
  esac
done

MODEL_DIR="${HOME}/bluejackets/models/${SLURM_JOB_ID}"
mkdir -vp "${MODEL_DIR}"

echo "Training: ${TRAINING_FILE}"
echo "Model: ${MODEL_DIR}"
echo

cd "${HTR_DIR}/src" || exit 99

COMMAND="python3 main.py ${OPTIONS} --data_file ${TRAINING_FILE} --model_dir ${MODEL_DIR}"
echo "${COMMAND}"
echo
${COMMAND}

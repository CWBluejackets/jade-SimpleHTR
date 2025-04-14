#!/bin/bash

#SBATCH --job-name=CWB_apply
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.harvey@sheffield.ac.uk

# load the modules for the program we want to run
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2019.07
# module load cuda/10.1
echo "modules loaded"

HTR_DIR="${HOME}/bluejackets/jade-SimpleHTR"

source activate simplehtr1
echo "environment activated"
echo

# OPTIONS passed to SimpleHTR
# -c convert fractions
# -d data control file
# -m model subdirectory
# -w wide_mode

OPTIONS="--mode apply "

while getopts "cwd:m:L:E:" option
do
  case "${option}" in
    c)
      OPTIONS="${OPTIONS} --convert_fractions"
      ;;
    w)
      OPTIONS="${OPTIONS} --wide_mode"
      ;;
    d)
      echo "${OPTARG}"
      X="$(realpath "${OPTARG}")"
      echo "${X}"
      DATA=$(realpath "${OPTARG}")
      ;;
    m)
      MODEL="${OPTARG}"
      ;;
    L)
      # learning rate
      OPTIONS="${OPTIONS} --opt_lr ${OPTARG}"
      ;;
    E)
      # epsilon
      OPTIONS="${OPTIONS} --opt_e ${OPTARG}"
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
echo "Word list: ${DATA}"
echo

source activate simplehtr1
echo "environment activated"
echo
cd "${HTR_DIR}/src" || exit 99

COMMAND="srun python3 main.py ${OPTIONS} --img_list_file ${DATA} --model_dir ${MODEL_DIR} --output_file ${OUTPUT_FILE}"
echo "${COMMAND}"
echo
${COMMAND}

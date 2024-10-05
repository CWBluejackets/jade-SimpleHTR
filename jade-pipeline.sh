#!/usr/bin/env bash
# Request 5 gigabytes of real memory (mem)
#$ -l rmem=5G

# set memory, nodes, GPUs, time limit
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=small
# time limit on big is 24h <http://docs.jade.ac.uk/en/latest/jade/scheduler/>

#SBATCH --job-name=CWB_pipeline
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

CONFIG_FILE="${HTR_DIR}/pipeline-config.json"
LOCATIONS_FILE="${HTR_DIR}/locations-jade.json"
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

COMMAND="python3 muster_pipeline.py ${OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}
echo

echo "About to create Zooniverse manifest"
echo

cd "${HTR_DIR}" || exit 98
MF_OPTIONS="-l ${LOCATIONS_FILE}"
COMMAND="python3 create_zoon_manifest.py ${MF_OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}
echo

#!/bin/bash

#SBATCH --job-name=CWB_pipeline
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.harvey@sheffield.ac.uk

# INSTRUCTIONS
# https://docs.hpc.shef.ac.uk/en/latest/bessemer/software/apps/python.html
# https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#job-submission-control-on-bessemer
# https://docs.hpc.shef.ac.uk/en/latest/bessemer/software/apps/python.html#python-conda-bessemer

# load the modules for the program we want to run
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2019.07
# module load cuda/10.1
echo "modules loaded"

HTR_DIR="${HOME}/bluejackets/jade-SimpleHTR"

# https://www.golinuxcloud.com/bash-getopts/
# https://www.computerhope.com/unix/bash/getopts.htm

CONFIG_FILE="${HTR_DIR}/pipeline-config.json"
LOCATIONS_FILE="${HTR_DIR}/locations-bessemer.json"
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

COMMAND="srun python3 muster_pipeline.py ${OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}
echo

echo "About to create Zooniverse manifest"
echo

cd "${HTR_DIR}" || exit 98
MF_OPTIONS="-l ${LOCATIONS_FILE}"
COMMAND="srun python3 create_zoon_manifest.py ${MF_OPTIONS}"
echo "${COMMAND}"
echo
${COMMAND}
echo

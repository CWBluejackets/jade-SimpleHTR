#!/bin/bash

#SBATCH --job-name=CWB_pipeline
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.funk@sheffield.ac.uk

# INSTRUCTIONS
# https://docs.hpc.shef.ac.uk/en/latest/bessemer/software/apps/python.html
# https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#job-submission-control-on-bessemer
# https://docs.hpc.shef.ac.uk/en/latest/bessemer/software/apps/python.html#python-conda-bessemer

# load the modules for the program we want to run
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2019.07
# module load cuda/10.1
echo "modules loaded"

working_directory="${HOME}/bluejackets/jade-SimpleHTR"

source activate simplehtr1
echo "environment activated"
echo

cd "${working_directory}" || exit 99

python3 adjudicate.py \
        -c config-bessemer.json \
        -f filters.json


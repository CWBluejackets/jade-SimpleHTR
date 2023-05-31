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

#SBATCH --job-name=adjudicate
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.funk@sheffield.ac.uk

# INSTRUCTIONS
# http://docs.jade.ac.uk/en/latest/jade/scheduler/index.html

# load the modules for the program we want to run
module load python/anaconda3
module load cuda/10.1
echo "modules loaded"

working_directory="${HOME}/bluejackets/jade-SimpleHTR"

source activate simplehtr1
echo "environment activated"
echo

cd "${working_directory}" || exit 99

python3 adjudicate.py \
        -c config-jade2.json \
        -f filters.json


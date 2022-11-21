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

# htr_directory="${HOME}/bluejackets/SimpleHTR"
htr_directory=$(dirname "$0")

source activate simplehtr1
echo "environment activated"

cd "${htr_directory}/src" || exit 99

# command line arguments:
# 1 = model subdirectory
# 2 = data subdirectory
# e.g. '22014_0.50'
# i.e., take model trained on $1 and test it on $2

model_directory="${HOME}/bluejackets/models/$1"
data_directory="${HOME}/bluejackets/training_data/$2"

echo "Model: ${model_directory}"
echo "Test data: ${data_directory}"

# create if new; empty out if re-used
rm -fr "${model_directory}"
mkdir -vp "${model_directory}"
echo

python3 main.py --mode validate \
      --data_dir "${data_directory}"  \
      --model_dir "${model_directory}"

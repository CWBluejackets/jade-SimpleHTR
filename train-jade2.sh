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

htr_directory="${HOME}/bluejackets/SimpleHTR"

source activate simplehtr1

echo "environment activated"

echo ; echo "PATH:"
echo "${PATH}"

echo ; echo "PYTHON"
which python
which python3

echo ; echo
cd "${htr_directory}/src" || exit 99

# "fast image loading" from the documentation
#rm -fr "${training_directory}/lmdb"
#python create_lmdb.py --data_dir "${training_directory}"

# get the workflow from the command line argument
# 22014 age
# 22038 term of enlistment

training_directory="${HOME}/bluejackets/training_data/$1"

python3 main.py --mode train \
      --data_dir "${training_directory}"  \
      --batch_size 500 --early_stopping 15


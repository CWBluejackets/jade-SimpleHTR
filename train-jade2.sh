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

htr_directory="${HOME}/bluejackets/jade-SimpleHTR"

source activate simplehtr1
echo "environment activated"

echo ; echo "PATH:"
echo "${PATH}"

echo ; echo "PYTHON"
which python
which python3

echo ; echo
cd "${htr_directory}/src" || exit 99

# get the workflow training subdirectory from the command line argument
# 22014 age
# 22038 term of enlistment
# now with sampling '22014_0.50'

# TODO use getopts for flags
# https://www.golinuxcloud.com/bash-getopts/
# https://www.computerhope.com/unix/bash/getopts.htm
# high contrast
# convert fractions
# modify the model directories' names accordingly?

training_directory="${HOME}/bluejackets/training_data/$1"
model_directory="${HOME}/bluejackets/models/$1"
debug_directory="${HOME}/bluejackets/debug"

echo "Training: ${training_directory}"
echo "Model: ${model_directory}"
echo

# create if new; empty out if re-used
rm -fr "${model_directory}"
mkdir -vp "${model_directory}"
rm -fr "${debug_directory}"
mkdir -vp "${debug_directory}"
echo

if [ "$2" = 'h' ]
then
    echo "high contrast"
    python3 main.py --mode train  \
            --high_contrast  --debug_dir "${debug_directory}"  \
            --data_dir "${training_directory}"    \
            --model_dir "${model_directory}"      \
            --batch_size 500 --early_stopping 15
else
    echo "normal contrast"
    python3 main.py --mode train  \
            --data_dir "${training_directory}"    \
            --model_dir "${model_directory}"      \
            --batch_size 500 --early_stopping 15
fi



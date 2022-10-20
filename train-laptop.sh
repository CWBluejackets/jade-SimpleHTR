#!/usr/bin/env bash
# default shell on jade2 is also bash
# ``echo $0` or `ps -p $$` in terminal to check

# age subdirectory
main_training_directory="/opt/data/bluejackets/training_data"
htr_directory="${HOME}/sandboxes/SimpleHTR"
main_model_directory="/opt/data/bluejackets/models"

# get the workflow training subdirectory from the command line argument
# 22014 age
# 22038 term of enlistment
# now with sampling '22014_0.50'


# clear out before running?
# model_directory="${htr_directory}/model"

# jade2 version will need to load modules

cd "${htr_directory}/src" || exit 99

# "fast image loading" from the documentation
# rm -fvr "${training_directory}/lmdb"
# conda run -n simplehtr1 python create_lmdb.py --data_dir "${training_directory}"

# https://stackoverflow.com/questions/70512216/conda-activate-in-script
# conda in a script
# conda run -n simplehtr python -c 'import sys; print(sys.path)'

#conda run -n simplehtr1 python main.py --mode train --fast \
#      --data_dir "${training_directory}"  \
#      --batch_size 50 --early_stopping 15

# clear out the existing model
rm -fv "${htr_directory}/model/*"


training_directory="${main_training_directory}/$1"
model_directory="${main_model_directory}/$1"

echo "Training: ${training_directory}"
echo "Model: ${model_directory}"

# create if new; empty out if re-used
rm -fvr "${model_directory}"
mkdir -vp "${model_directory}"
echo

conda run -n simplehtr1 python main.py --mode train \
      --data_dir "${training_directory}"  \
      --model_dir "${model_directory}" \
      --batch_size 100 --early_stopping 15

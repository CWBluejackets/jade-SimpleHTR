#!/usr/bin/env bash
# default shell on jade2 is also bash
# ``echo $0` or `ps -p $$` in terminal to check

training_directory="/opt/data/bluejackets/training_data"
htr_directory="${HOME}/sandboxes/SimpleHTR"

# clear out before running?
# model_directory="${htr_directory}/model"

mkdir -p "${training_directory}/lmdb"

# jade2 version will need to load modules

cd "${htr_directory}/src" || exit 99

# https://stackoverflow.com/questions/70512216/conda-activate-in-script
# conda in a script

conda run -n simplehtr python main.py --mode train --fast \
      --data_dir "${training_directory}"  \
      --batch_size 500 --early_stopping 15

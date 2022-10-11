#!/usr/bin/env bash
# default shell on jade2 is also bash
# ``echo $0` or `ps -p $$` in terminal to check

training_directory="/opt/data/bluejackets/training_data"
htr_directory="${HOME}/sandboxes/SimpleHTR"

# clear out before running?
# model_directory="${htr_directory}/model"

# jade2 version will need to load modules

conda activate simplehtr

cd "${htr_directory}/src" || exit 99

python  --mode train --fast --data_dir "${training_directory}"  --batch_size 500 --early_stopping 15

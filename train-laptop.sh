#!/usr/bin/env bash
# default shell on jade2 is also bash
# ``echo $0` or `ps -p $$` in terminal to check

# age subdirectory
training_directory="/opt/data/bluejackets/training_data/22014"
htr_directory="${HOME}/sandboxes/SimpleHTR"

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

conda run -n simplehtr1 python main.py --mode train \
      --data_dir "${training_directory}"  \
      --batch_size 100 --early_stopping 15

#!/usr/bin/env bash
# default shell on jade2 is also bash
# ``echo $0` or `ps -p $$` in terminal to check

main_model_directory="/opt/data/bluejackets/models"

htr_directory="${HOME}/sandboxes/bluejackets-jade-SimpleHTR"

cd "${htr_directory}/src" || exit 1

if [ -z "$1" ]
then
  echo "Both arguments missing!"
  exit 2
elif [ -z "$2"  ]
then
  echo "One argument missing!"
  exit 5
fi


# https://stackoverflow.com/questions/70512216/conda-activate-in-script
# conda in a script
# conda run -n simplehtr python -c 'import sys; print(sys.path)'

# clear out the existing model

training_file="$1"
model_directory="${main_model_directory}/$2"

echo "Training: ${training_file}"
echo "Model: ${model_directory}"

# create if new; empty out if re-used
rm -fvr "${model_directory}"
mkdir -vp "${model_directory}"
echo

conda run -n simplehtr1 python main.py --mode train \
      --data_file "${training_file}"  \
      --model_dir "${model_directory}" \
      --batch_size 100 --early_stopping 15

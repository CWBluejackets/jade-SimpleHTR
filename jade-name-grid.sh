#!/usr/bin/env bash

# sbatch jade-SimpleHTR/jade-train.sh -a  -t training_data/21955/words_0.25.txt

for RATE in  0.001  0.002  0.003  0.004  0.005
do
  for EPSILON in  1e-1  1e-2  1e-3  1e-4  1e-5  1e-6  1e-7  1e-8
  do
    echo
    COMMAND="Sbatch jade-SimpleHTR/jade-train.sh -a -L ${RATE} -E ${EPSILON}  -t training_data/21955/words_0.25.txt"
    echo "${COMMAND}"
    ${COMMAND}
    echo
  done
done

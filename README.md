# jade-SimpleHTR
conda set-up and scripts for using SimpleHTR on JADE2

## Python environment

```bash
conda create -n htr2 python==3.10
conda activate htr2
pip install -r requirements.txt
```

## batch submission on jade2

This does *not* work if the environment is active in the shell when you run `sbatch`!

## Config file

* `input_directory` containing CSV files from Zooniverse
* `intermediate_file` our CSV format for generating training data
* `jpg_directory` contains the JPGs that the annotators use, required
* `big_jpg_directory` contains the full-size JPGs, optional
* `training_directory` training datasets will be created here in the IAM format
* `log_directory` a timestamped log file will go here
* `suppress_spaces` delete whitespace, boolean
* `sampling` generate four sizes of training datasets (25%, 50%, 75%, 100% of what is available)

## Filter file (WIP)

Key-value pairs: workflow (`str` ) to regex (`str`).
If a workflow in the data does not have a filter specified, 
everything is passed through.

Note: the regex will be applied using `fullmatch`.

## new training data structure

### File structure

info/annotations_000000085.csv
img/0e0f9b6f3bf64dd680c8c033cb73a9c0_0000085.png
info/annotations_000000124.csv
img/0e0f9b6f3bf64dd680c8c033cb73a9c0_0000124.png

```
.
├── words.txt
├── word_count.tsv
├── info/
    ├── 0e0f9b6f3bf64dd680c8c033cb73a9c0_0000085.csv
    ├── 0e0f9b6f3bf64dd680c8c033cb73a9c0_0000124.csv
    ...
└── img
    ├── 0e0f9b6f3bf64dd680c8c033cb73a9c0_0000085.png
    ├── 0e0f9b6f3bf64dd680c8c033cb73a9c0_0000124.png
    ...
```

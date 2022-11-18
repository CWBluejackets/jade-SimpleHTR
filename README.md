# jade-SimpleHTR
conda set-up and scripts for using SimpleHTR on JADE2

Target:
https://github.com/githubharald/SimpleHTR

## Python environment

```bash
conda create -n simplehtr1 python==3.7
conda activate simplehtr1
cd path/to/SimpleHTR
pip install -r requirements.txt
```

(The scripts use pandas and pillow; the rest is from SimpleHTR's 
requirements file.)

### Conda note

* conda `lmbd` package: the c libraries for the database application itself.
* conda `python-lmdb` package: provides python module lmdb
* might not be relevant using pip?

## batch submission on jade2

This does *not* work if the environment is active in the shell when you run `sbatch`!

## Config file

* `input_directory` containing CSV files from Zoonivers
* `intermediate_file` our CSV format for generating training data
* `jpg_directory` contains the JPGs that the annotators use, required
* `big_jpg_directory` contains the full-size JPGs, optional
* `training_directory` training datasets will be created here in the IAM format
* `log_directory` a timestamped log file will go here
* `suppress_spaces` delete whitespace, boolean
* `digits_only` ignore annotations containing anything other than digits, boolean
* `sampling` generate four sizes of training datasets (25%, 50%, 75%, 100% of what is available)

## Filter file (WIP)

Key-value pairs: workflow (`str` will be parsed to `int`) to regex (`str`).
If a workflow in the data does not have a filter specified, 
everything is passed through.

Note: the regex will be applied using `fullmatch`.

## new training data structure

### File structure

```
.
├── words.txt
├── info/
    ├── jsalfdkjlsd.info
    ├── ljsfdkjs.info
    ...
└── img
    ├── jsalfdkjlsd.png
    ├── ljsfdkjs.png
    ...
```

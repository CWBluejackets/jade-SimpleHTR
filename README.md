# jade-SimpleHTR plus muster pipeline

## jade-SimpleHTR training and testing

conda set-up and scripts for using SimpleHTR on JADE2

Target:
https://github.com/githubharald/SimpleHTR

### Python environment

```bash
conda create -n simplehtr1 python==3.7
conda activate simplehtr1
cd path/to/SimpleHTR
pip install -r requirements.txt
```

(The scripts use pandas and pillow; the rest is from SimpleHTR's 
requirements file.)

#### Conda note

* conda `lmbd` package: the c libraries for the database application itself.
* conda `python-lmdb` package: provides python module lmdb
* might not be relevant using pip?

### batch submission on jade2

This does *not* work if the environment is active in the shell when you run `sbatch`!

### Config file

* `input_directory` containing CSV files from Zooniverse
* `intermediate_file` our CSV format for generating training data
* `jpg_directory` contains the JPGs that the annotators use, required
* `big_jpg_directory` contains the full-size JPGs, optional
* `training_directory` training datasets will be created here in the IAM format
* `log_directory` a timestamped log file will go here
* `suppress_spaces` delete whitespace, boolean
* `digits_only` ignore annotations containing anything other than digits, boolean
* `sampling` generate four sizes of training datasets (25%, 50%, 75%, 100% of what is available)

### Filter file (WIP)

Key-value pairs: workflow (`str` ) to regex (`str`).
If a workflow in the data does not have a filter specified, 
everything is passed through.

Note: the regex will be applied using `fullmatch`.

### new training data structure

#### File structure

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

## muster pipeline

Pipeline for processing muster JPGs using trained models.

### Tools

* to_name = read JPGs; write control files and lots of PNGs
* to_name = read control file then PNGs; apply models; write CSV (probably)
* **TODO** add API to jade-SimpleHTR for applying models to PNGs in bulk

### Input information

* JPG filename (or input directory for bulk processing on server)
* muster date if present --- might be easier for a human to do this since
  we haven't been training on it
* muster formats (in a file referenced by the config file)
  * columns to be merged (e.g., eyes + hair + complexion; feet + inches)
  * margin hints?
  * typical row heights
* config file

### Derived information

* from filename
  * vessel name
  * sequence number
* from box detection
  * column numbers for reassembly
  * row numbers for reassembly
  * column headings / models
* from ML application
  * label
  * confidence score

### Configuration ideas

* input directory (config file for server or CL options for testing)
* parameters, for rows and columns separately
  * Canny min and max thresholds (TODO figure out what these mean)
    * if gradient > high threshold it's a "strong" edge pixel
    * if gradient < low threshold it's suppressed
    * The two threshold values are empirically determined and their definition will depend on the content of a given input image.
      https://en.wikipedia.org/wiki/Canny_edge_detector#Double_threshold
  * Canny aperture (5 works better than default 3)
  * Hough threshold
  * Hough min length
  * Hough max gap
  * deviation (degrees from vertical/horizontal)
  * maximum H/V distance between V/H lines to group and merge
  * minimum number of lines to constitute a valid group
  * merging function (H max, V mean)
* confidence score required to use the automatic annotation without checking
  (0.0 to 1.0)
* directory for images and labels to be checked and corrected
  * How to format image-label combination for Zooniverse?
* intermediate directory: PNGs, maybe control files
* hints for muster format/structure
* output directory: CSVs

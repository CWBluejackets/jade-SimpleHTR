# bessemer notes

## conda environment

```shell
srun --pty bash -i 
module load Anaconda3/2025.06-1
conda init # ONLY NEEDED THE FIRST TIME
conda create -n simplehtr1 python=3.9
conda activate simplehtr1
cd bluejackets/jade-SimpleHTR/
pip install -r requirements.txt
```

Note: `requirements.txt` required some modification to include the torch package and a numpy version compatible with the out-of-date tensorflow.

## locations and running

- models: `/home/cs1jafu/bluejackets/models/<slurm_id>`
  - ID from slurm job number when training
  - these are set in `pipeline-models-bessemer.json`
- pipeline
  - input: image files from `${HOME}/bluejackets/pipeline/input/` 
  - `${HOME}/bluejackets/pipeline/output/` contains 
    - the registration files `*.png`
    - the HTR and blank detection output `*.csv`
    - the debugging information `*.json`
  - `${HOME}/bluejackets/pipeline/corrections/json/` contains `*.json` individual metadata files
  - `${HOME}/bluejackets/pipeline/corrections/image/` contains
    - `*.jpg` image files for Zooniverse phase 2
    - `manifest.json` compiled from the individual metadata files
- uploading for phase 2
  - input: the `corrections/image/` directory above
  - `phase-2-make-batches.py` splits the `manifest.json` file into separate batches (by vessel-year and 1000s of items; each will become one subject set on Zooniverse)
  - `phase-2-upload-batches.py` takes each manifest batch file, uploads the subjects to Zooniverse, and saves a modified manifest with the subject ID 
  - This does not add the new subject sets to the phase 2 workflow; Damian usually does that. 

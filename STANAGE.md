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




```shell
srun --pty bash -i 
module load Anaconda3/2025.06-1
conda init
conda create -n simplehtr1 python=3.9
conda activate simplehtr1
cd bluejackets/jade-SimpleHTR/

# not sure if following was necessary
pip install numpy matplotlib opencv-python
pip install scipy
pip install lmdb==1.0.0
pip install path==15.0.0
pip install scikit-image

pip install -r requirements.txt
```

### results

```shell
$ conda list |egrep 'tf|tensor'
tensorboard                     2.14.1           pypi_0           pypi
tensorboard-data-server         0.7.2            pypi_0           pypi
tensorflow                      2.14.0           pypi_0           pypi
tensorflow-estimator            2.14.0           pypi_0           pypi
tensorflow-io-gcs-filesystem    0.37.1           pypi_0           pypi
```

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

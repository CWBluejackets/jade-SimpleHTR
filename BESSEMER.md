# bessemer notes

## conda environment

```shell
srun --pty bash -i 
module load Anaconda3/2019.07
conda create -n simplehtr1 python=3.8
conda activate simplehtr1
cd bluejackets/jade-SimpleHTR/
pip install -r requirements.txt
```

- can't find tensorflow 2.14
- tried python 3.7 and 3.6, no luck
- https://stackoverflow.com/questions/41937915/how-to-pip-install-old-version-of-librarytensorflow
- https://pypi.org/project/tensorflow/2.14.1/#files
- no wheel for python below 3.9

```shell
conda install python=3.9
export TF_BINARY_URL=https://files.pythonhosted.org/packages/18/19/50a32e3017b6ba2e51a1c9d717e2f1e0e4edde5b5fdcf030ec5a19eb6ba2/tensorflow-2.14.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install $TF_BINARY_URL
pip install -r requirements.txt 
# version incompatibilities with three packages so...
pip install numpy matplotlib opencv-python
pip install scipy
pip install scikit-image==0.19
pip install lmdb==1.0.0
pip install path==15.0.0
pip install levenshtein editdistance pandas
```

## locations

- training data: `/fastdata/li1af/bluejackets/training_data/<workflow_id>`
  - ID from original workflows
- models: `/home/li1af/bluejackets/models/<slurm_id>`
  - ID from slurm job number when training

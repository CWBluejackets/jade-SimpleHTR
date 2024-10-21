# bessemer notes

## conda environment

```shell
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
# and hope for the best
```

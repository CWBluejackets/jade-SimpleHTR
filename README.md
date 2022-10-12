# jade-SimpleHTR
conda set-up and scripts for using SimpleHTR on JADE2

Target:
https://github.com/githubharald/SimpleHTR

so far:
```bash
conda create -n simplehtr python==3.9
conda activate simplehtr
conda install -c conda-forge editdistance lmdb matplotlib numpy path tensorflow
pip install opencv-python
conda install -c conda-forge python-lmdb
```

## Python environment

```bash
conda create -n simplehtr1 python==3.7
pip install -r requirements.txt
pip install pandas pillow
```

(The scripts use pandas and pillow.)
 
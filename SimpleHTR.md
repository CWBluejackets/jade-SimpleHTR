# Using SimpleHTR

```bash
conda create -n simplehtr python==3.9
conda activate simplehtr
conda install -c conda-forge editdistance lmdb matplotlib numpy path tensorflow
pip install opencv-python
conda install -c conda-forge python-lmdb
```

https://github.com/ContinuumIO/anaconda-issues/issues/8740

lmbd = the c libraries for the database application itself.
The python package "lmdb" is actually in the conda package "python-lmdb".

## training data

http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

## format

```
#     a01-000u-00-00  -> word id for line 00 in form a01-000u
#     ok              -> result of word segmentation
#                            ok: word was correctly
#                            er: segmentation of word can be bad
#
#     154             -> graylevel to binarize the line containing this word
#     1               -> number of components for this word
#     408 768 27 51   -> bounding box around this word in x,y,w,h format
#     AT              -> the grammatical tag for this word, see the
#                        file tagset.txt for an explanation
#     A               -> the transcription for this word
#
a01-000u-00-00 ok 154 408 768 27 51 AT A
```
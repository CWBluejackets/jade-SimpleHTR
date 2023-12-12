# pipeline

Pipeline for processing muster JPGs using trained models.

## Tools

* to_name = read JPGs; write control files and lots of PNGs
* to_name = read control file then PNGs; apply models; write CSV (probably)
* **TODO** add API to jade-SimpleHTR for applying models to PNGs in bulk

## Input information

* JPG filename
* muster date if present --- might be easier for a human to do this
* typical muster formats (in a file referenced by the config file)
* config file

## Derived information

* from filename
  * vessel name
  * sequence number
* from content
  * columns
  * rows
  * column headings / models



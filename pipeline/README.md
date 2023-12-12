# pipeline

Pipeline for processing muster JPGs using trained models.

## Tools

* to_name = read JPGs; write control files and lots of PNGs
* to_name = read control file then PNGs; apply models; write CSV (probably)
* **TODO** add API to jade-SimpleHTR for applying models to PNGs in bulk

## Input information

* JPG filename (or input directory for bulk processing on server)
* muster date if present --- might be easier for a human to do this
* muster formats (in a file referenced by the config file)
  * columns to be merged (e.g., eyes + hair + complexion; feet + inches) 
* config file

## Derived information

* from filename
  * vessel name
  * sequence number
* from content
  * column numbers for reassembly
  * row numbers for reassembly
  * column headings / models

## Configuration ideas

* input directory (config file for server or CL options for testing)
* intermediate directory: PNGs, maybe control files
* hints for muster format/structure
* output directory: CSVs

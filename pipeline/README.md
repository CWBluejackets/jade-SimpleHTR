# pipeline

Pipeline for processing muster JPGs using trained models.

## Tools

* to_name = read JPGs; write control files and lots of PNGs
* to_name = read control file then PNGs; apply models; write CSV (probably)
* **TODO** add API to jade-SimpleHTR for applying models to PNGs in bulk

## Input information

* JPG filename (or input directory for bulk processing on server)
* muster date if present --- might be easier for a human to do this since
  we haven't been training on it
* muster formats (in a file referenced by the config file)
  * columns to be merged (e.g., eyes + hair + complexion; feet + inches)
  * margin hints?
  * typical row heights
* config file

## Derived information

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

## Configuration ideas

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

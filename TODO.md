# TODO

## running now

* double up the term (22038) training data and train over it
* combine the term and age (22014) training data into one set and train

## further ideas

* change the SHTR training dataset to be easier to generate and manipulate
  * simplify words.txt format
  * flatten the PNG file structure
* preprocessing ideas
  * add perturbed versions to the training data 
    * without replacing the plain versions
    * to be done outside of SHTR
  * increase contrast by min-max normalization of pixel values to 0--255
    * this could be done inside the SHTR code for easy consistency between tRaining and application
* look at the hyperparameters, particularly for tuning the gradient descent
* apply the age model to the term data and check for accuracy
 
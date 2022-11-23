# TODO

## evaluation

* fix occupation filter to include apostrophes and ditto marks
* add merge age+term command to `create-training-jade2.sh`

## further ideas

* preprocessing ideas
  * add perturbed versions to the training data 
    * without replacing the plain versions
    * to be done outside of SHTR
  * increase contrast by min-max normalization of pixel values to 0--255
    * this could be done inside the SHTR code for easy consistency between tRaining and application
* look at the hyperparameters, particularly for tuning the gradient descent
* apply the age model to the term data and check for accuracy
 
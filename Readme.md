Trains a three layer neural network to recognize hand written digits using the [MNIST](http://yann.lecun.com/exdb/mnist/) data set.

Note: trainData.mat and testData.mat used by the code are not included here due to the size of these files.  The readInData.m script can be used create these files from the files available at http://yann.lecun.com/exdb/mnist/.

train.m will train a network on the training set and store the weights in the weights.mat.

test.m will apply the network to the test set and print out the precentage of images that were correctly classified.

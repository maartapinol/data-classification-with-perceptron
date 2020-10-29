# data-classification-with-perceptron

Perceptron is a supervised algorithm for binary classification i.e. elements are classified in two classes. Supervised means that in this type of classification we initially have knowledge about the category or class to which each of the elements of the training data set belong. 


Perceptron uses the following linear prediction function as a classifier of the input vectors in two classes or labels:

![equation](https://bit.ly/3mw41ee)



w is the vector of weights, x the input vector, and b, the intercept of the linear function, is called the bias.

The Perceptron algorithm, as all the supervised techniques, consists of two phases:

•	Training phase: we take a subset of data from our dataset with which we are going to train our model so that it "learns to predict".

•	Testing phase: we take the remaining data subset from our dataset and with it we check that our previously trained model is able to correctly predict new observations that have not appeared in the training data set.



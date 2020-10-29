# data-classification-with-perceptron

Perceptron is a supervised algorithm for binary classification i.e. elements are classified in two classes. Supervised means that in this type of classification we initially have knowledge about the category or class to which each of the elements of the training data set belong. 


Perceptron uses the following linear prediction function as a classifier of the input vectors in two classes or labels:

![equation](http://www.sciweavers.org/tex2img.php?eq=%20f%28x%29%20%3D%5Cbegin%7Bcases%7D1%20%26%20%20w%5E%7BT%7Dx%20%2B%20b%20%20%5Cgeq%20%200%5C%5C-1%20%26%20w%5E%7BT%7Dx%20%2B%20b%20%3C%200%20%5Cend%7Bcases%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

![equation]([img]http://www.sciweavers.org/tex2img.php?eq=w&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0[/img]) is the vector of weights

The Perceptron algorithm, as all the supervised techniques, consists of two phases:
•	Training phase: we take a subset of data from our dataset with which we are going to train our model so that it "learns to predict".
•	Testing phase: we take the remaining data subset from our dataset and with it we check that our previously trained model is able to correctly predict new observations that have not appeared in the training data set.

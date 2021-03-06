import numpy as np
import pandas as pd


def PerceptronTraining(TrainingData, labelVar, class1, class2, MaxNumIter):

    ### Pre training ###
    classes = class1+class2 # classes to discriminate
    TrainingData = TrainingData[TrainingData[labelVar].isin(classes)] # filter the df by the classes

    TrainingDataX = TrainingData.drop(labelVar, axis=1) # df for features
    TrainingDataY = TrainingData[labelVar] # original labels
    TrainingDataY = np.where(TrainingDataY.isin(class1), 1, -1) # set one class to 1 and the other to -1
    
    ### Initialization ###
    weights=[0]*len(TrainingDataX.columns) # initialize weights to 0
    bias = 0 # initialize bias to 0
    
    ### Iterations ###
    for iter in list(range(MaxNumIter)):
        
        for i in list(range(len(TrainingDataX))):
            a = sum(weights*TrainingDataX.iloc[i]) + bias # compute activation
            
            if float(TrainingDataY[i]*a) <= 0: 
                weights = weights + TrainingDataY[i]*TrainingDataX.iloc[i] # update weights
                bias = bias + TrainingDataY[i] # update bias
    
    sign_a = np.sign(a) # compute sign of the activation values
    accuracy = sum(TrainingDataY == sign_a)/(sum(TrainingDataY == sign_a) + sum(TrainingDataY != sign_a)) # compute accuracies
    
    return weights, bias, accuracy

def PerceptronTesting(TestData, labelVar, class1, class2, weights, bias):

    ### Pre training ###
    classes = class1+class2 # classes to discriminate
    TestData = TestData[TestData[labelVar].isin(classes)] # filter the df by the classes

    TestDataX = TestData.drop(labelVar, axis=1) # df for features
    TestDataY = TestData[labelVar] # original labels
    TestDataY = np.where(TestDataY.isin(class1), 1, -1) # set one class to 1 and the other to -1
    
    a = []
    
    ### Iterations ###
    for i in list(range(len(TestDataX))):
        a = a + [sum(weights*TestDataX.iloc[i]) + bias] # compute activation for the test

    sign_a = np.sign(a) # compute sign of the activation values

    accuracy = sum(TestDataY == sign_a)/(sum(TestDataY == sign_a) + sum(TestDataY != sign_a)) # compute accuracies

    return accuracy


traindf = pd.read_table("train.data", header = 0, sep = ",")
testdf = pd.read_table("test.data", header = 0, sep = ",")

Iter = 20
y = "y"
class1 = ["class-3"]
class2 = ["class-1","class-2"]

w,b,train_accuracy = PerceptronTraining(traindf, y, class1, class2, Iter)
print("Weights: \n", w, "\nbias:", b, "\nTrain Classification Accuracy:", train_accuracy)

test_accuracy = PerceptronTesting(testdf, y, class1, class2, w, b)
print("Test Classification Accuracy: ", test_accuracy)

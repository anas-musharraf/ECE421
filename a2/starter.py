import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def reshape_data_tensor(tensor):
    arg1,arg2,arg3 = tensor.shape
    new_tensor = tensor.reshape((arg1, (arg2*arg3)))
    return new_tensor

def reshape_target_tensor(tensor):
    new_tensor = tensor.reshape(tensor.shape[0], 1)
    return new_tensor
    
def relu(x):
    #alternate method x*(x>0)
    return np.maximum(x,0)
    

def softmax(x):
    a = np.exp(x-np.max(x))
    return a/a.sum()



def computeLayer(X, W, b):
    return np.matmul(X,W) + b

def CE(target, prediction):
    interim = target * np.log(prediction)
    interim2 = np.sum(interim, axis=1)
    interim3 = np.sum(interim2, axis=0)
    return (-1/target.shape[0])*interim3


def gradCE(target, prediction):
    return softmax(prediction) - target

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = reshape_data_tensor(trainData)
validData = reshape_data_tensor(validData)
testData = reshape_data_tensor(testData)
trainTarget = reshape_target_tensor(trainTarget)
validTarget= reshape_target_tensor(validTarget)
testTarget = reshape_target_tensor(testTarget)
newTrain, newValid, newTest = convertOneHot(trainTarget, validTarget, testTarget)


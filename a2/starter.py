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


#1.2 Functions

def gradLossOuterWeight(target,prediction,h):
    grad = gradCE(target,prediction)
    return np.matmul(np.transpose(h), grad)
    
def gradLossOuterBias(target,prediction):
    return gradCE(target,prediction)

def gradLossHiddenWeights(target,prediction,X,W):
    grad = gradCE(target,prediction)
    interim = np.matmul(np.transpose(X),grad)
    return np.matmul(interim, np.transpose(W))

def gradLossHiddenBiases(target,prediction,W):
    grad = gradCE(target,prediction)
    return np.matmul(grad, np.transpose(W))
    
def init_weight_vector(units_in, units_out):
    vector = np.random.normal(0, np.sqrt(2/units_in+units_out), (units_in, units_out))
    return vector
    
def learning(W_o, v_o, b_o, W_h, v_h, b_h, epochs, gamma, learningRate, cK, trainData, trainTarget):
    v_o_init = v_o
    b_o_init = b_o
    v_h_init = v_h
    b_h_init = b_h
    accuracy_train = []
    loss_train = []

    for i in range(epochs):
        z_hidden = computeLayer(trainData,W_h, b_h)
        a_hidden = relu(z_hidden)
        
        z_output = computeLayer(a_hidden, W_o, b_o)
        a_output = softmax(z_output)
        
    
    
def test_function():
    #CONSTANTS
    cLearningRate = 10^-5
    cGamma = 0.99
    cEpochs = 200
    cK = 1000
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = reshape_data_tensor(trainData)
    print(trainData.shape)
    validData = reshape_data_tensor(validData)
    testData = reshape_data_tensor(testData)
    trainTarget2 = reshape_target_tensor(trainTarget)
    validTarget2 = reshape_target_tensor(validTarget)
    testTarget2 = reshape_target_tensor(testTarget)
    newTrain, newValid, newTest = convertOneHot(trainTarget, validTarget, testTarget)
    
    W_h_init = init_weight_vector(trainData.shape[0], cK)
    v_h_init = np.full((trainData.shape[0], cK), 1E-7)
    b_h_init = np.zeros(1, cK)
    W_o_init = init_weight_vector(cK,10)
    v_o_init = np.full((cK, 10), 1E-7)
    b_o_init = np.zeros(1, 10)
    
    print(W_o_init)
    print(W_h_init)
    
    return 0

test_function()
    
    
    


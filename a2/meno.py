#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:24:18 2019

@author: anirudhsampath
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pdb


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

def relu(x):
    zeros = np.zeros(x.shape)
    return np.maximum(x, zeros)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims = True)


def computeLayer(X, W, b):
    return X@W + b

#assuming target is a one hot vector pertaining to which class it
#corresponds to
def CE(target, prediction):
    N = target.shape[0]
    return -1/N*np.sum(target*np.log(prediction))

def gradCE(target, prediction):
    N = target.shape[0]
    np.apply_along_axis(softmax, 0, prediction)
    return (-1/N)*np.sum(np.multiply(target,(1/prediction)))

def relu_backward(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

def init(hidden_nodes):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    trainData, trainTarget = shuffle(trainData, trainTarget)

    W_h = np.random.normal(0,np.sqrt(2/(784 + hidden_nodes)),(784, hidden_nodes))
    W_o = np.random.normal(0,np.sqrt(2/(hidden_nodes+10)),(hidden_nodes,10))

    b_h = np.random.normal(0,np.sqrt(2/(784 + hidden_nodes)),(1, hidden_nodes))
    b_o = np.random.normal(0,np.sqrt(2/(hidden_nodes+10)),(1,10))

    return trainData, validData, testData, trainTarget, validTarget, testTarget, W_h, b_h, W_o, b_o

########### initialization
hidden_nodes = 1000
output_labels = 10
epochs = 200
trainData, validData, testData, trainTarget, validTarget, testTarget, wh, bh, wo, bo = init(hidden_nodes)
feature_set = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
instances = feature_set.shape[0]
attributes = feature_set.shape[1]
lr = 1e-5
gamma = 0.99
error_cost = []

vwh = np.full((784, hidden_nodes), 1e-5)
vbh = np.full((1, hidden_nodes), 1e-5)
vwo = np.full((hidden_nodes, 10), 1e-5)
vbo = np.full((1, 10), 1e-5)
accuracy_train = []

for epoch in range(epochs):
############# feedforward

    # Phase 1
    zh = computeLayer(feature_set, wh, bh)
    ah = relu(zh)

    # Phase 2
    zo = computeLayer(ah, wo, bo)
    ao = softmax(zo)
    #pdb.set_trace()
########## Back Propagation

########## Phase 1

    dcost_dzo = (softmax(ao) - trainTarget)/instances
    dcost_wo = np.matmul(ah.T, dcost_dzo)
    dcost_bo = dcost_dzo.sum(axis=0).reshape(1,-1)


########## Phases 2

    dzo_dah = wo
    dcost_dah = dcost_dzo @ dzo_dah.T
    dah_dzh = relu_backward(zh)
    dzh_dwh = feature_set
    dcost_wh = dzh_dwh.T@(dcost_dah* dah_dzh)
    dcost_bh = (dcost_dah * dah_dzh).sum(axis=0).reshape(1,-1)

    # Update Weights ================

    vwh = gamma*vwh + lr*dcost_wh
    wh -= vwh
    vbh = (gamma*vbh + lr*dcost_bh.sum(axis=0))
    bh -= vbh

    vwo = gamma*vwo + lr*dcost_wo
    wo -= vwo
    vbo = (gamma*vbo + lr*dcost_bo.sum(axis=0))
    bo -= vbo
    #pdb.set_trace()
    #loss = np.sum((-trainTarget * np.log(ao)))/instances
    
    predict_result_matrix = np.argmax(ao, axis = 1)
    actual_result_matrix = np.argmax(trainTarget, axis=1)
    compare = np.equal(predict_result_matrix, actual_result_matrix)
    acc = (np.sum((compare==True))/(instances))
    accuracy_train.append(np.sum((compare==True))/(instances))
    loss = CE(trainTarget, ao)
    print("Epoch", epoch)
    print('Loss function value: ', loss)
    print("Accuracy", acc)
    error_cost.append(loss)
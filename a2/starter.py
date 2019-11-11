#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:52:05 2019

@author: anirudhsampath
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pdb
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
    #print(x[0][0])
    return np.maximum(x,0)
    

def softmax(x):
    a = np.exp(x-np.max(x))
    return a/a.sum()

def softmax2(x):
    #pdb.set_trace()
    max_of_row = np.amax(x,axis=1).reshape(x.shape[0],1)
    a = np.exp(x-max_of_row)
    return a/np.sum(a,axis=1, keepdims=True)



def computeLayer(X, W, b):
    return np.matmul(X,W) + b

def CE(target, prediction):
    interim = target * np.log(prediction)
    interim2 = np.sum(interim, axis=1)
    interim3 = np.sum(interim2, axis=0)
    return (-1/target.shape[0])*interim3


def gradCE(target, prediction):
    #return (1/(target.shape[0]))*(softmax2(prediction) - target)
    return softmax2(prediction)-target


#1.2 Functions

def gradLossOuterWeight(target,prediction,h):
    grad = gradCE(target,prediction)
    return np.matmul(np.transpose(h), grad)
    
def gradLossOuterBias(target,prediction):
    return np.sum(gradCE(target,prediction), axis=0).reshape(1,-1)

def gradLossHiddenWeights(target,prediction,X,W, z_hidden):
    backwardRelu(z_hidden)
    grad = gradCE(target,prediction)
    #print(grad.shape)
    #interim = np.matmul(np.transpose(X),grad)
    #abc = np.matmul(interim, np.transpose(W))
    interim = np.multiply(np.matmul(grad, np.transpose(W)),z_hidden)
    b = np.matmul(np.transpose(X), interim)
    #pdb.set_trace()
    return b
    
def gradLossHiddenBiases(target,prediction,W):
    grad = gradCE(target,prediction)
    return np.sum(np.matmul(grad, np.transpose(W)), axis=0).reshape(1,-1)
    
def backwardRelu(X):
    X[X<=0] = 0
    X[X>0] = 1
    return X
    
def init_weight_vector(units_in, units_out):
    vector = np.random.normal(0, np.sqrt(2/(units_in+units_out)), (units_in, units_out))
    return vector
    
def learning(W_o, v_o, b_o, W_h, v_h, b_h, epochs, gamma, learningRate, trainData, trainTarget, validData, validTarget, testData, testTarget):
    v_o_init = v_o
    b_o_init = b_o
    v_h_init = v_h
    b_h_init = b_h
    acc_train = []
    loss_train = []
    acc_valid = []
    loss_valid = []
    acc_test = []
    loss_test = []

    for i in range(epochs):
        #print(i)
        #pdb.set_trace()
        z_hidden = computeLayer(trainData,W_h, b_h)
        z_h_copy = z_hidden
        a_hidden = relu(z_hidden)
        
        z_output = computeLayer(a_hidden, W_o, b_o)
        a_output = softmax2(z_output)
        
        z_h_valid = computeLayer(validData,W_h, b_h)
        a_h_valid = relu(z_h_valid)
        z_out_valid = computeLayer(a_h_valid, W_o, b_o)
        a_out_valid = softmax2(z_out_valid)
        
        z_h_test = computeLayer(testData,W_h, b_h)
        a_h_test = relu(z_h_test)
        z_out_test = computeLayer(a_h_test, W_o, b_o)
        a_out_test = softmax2(z_out_test)
        
        v_o_init = gamma*v_o_init + learningRate*gradLossOuterWeight(trainTarget, z_output, a_hidden)
        b_o_init = gamma*b_o_init + learningRate*gradLossOuterBias(trainTarget, z_output)

        v_h_init = gamma*v_h_init + learningRate*gradLossHiddenWeights(trainTarget, z_output, trainData, W_o, z_h_copy)
        b_h_init = gamma*b_h_init + learningRate*gradLossHiddenBiases(trainTarget, z_output, W_o)
        
        W_o = W_o - v_o_init
        b_o = b_o - b_o_init
        W_h = W_h - v_h_init
        b_h_init = b_h - b_h_init
        #pdb.set_trace()
        #pdb.set_trace()
       # print(CE(trainTarget, a_output))
        predict_result_matrix = np.argmax(a_output, axis = 1)
        actual_result_matrix = np.argmax(trainTarget, axis=1)
        compare = np.equal(predict_result_matrix, actual_result_matrix)
        acc = (np.sum((compare==True))/(trainData.shape[0]))
        acc_train.append(np.sum((compare==True))/(trainData.shape[0]))
        loss_train.append(CE(trainTarget, a_output))
        
        pred_valid = np.argmax(a_out_valid, axis = 1)
        act_valid = np.argmax(validTarget, axis=1)
        compare_valid = np.equal(pred_valid, act_valid)
        acc_v = (np.sum((compare_valid==True))/(validData.shape[0]))
        acc_valid.append((np.sum((compare_valid==True))/(validData.shape[0])))
        loss_valid.append(CE(validTarget, a_out_valid))
        
        pred_test = np.argmax(a_out_test, axis = 1)
        act_test = np.argmax(testTarget, axis=1)
        compare_test = np.equal(pred_test, act_test)
        acc_t = (np.sum((compare_test==True))/(testData.shape[0]))
        acc_test.append((np.sum((compare_test==True))/(testData.shape[0])))
        loss_test.append(CE(testTarget, a_out_test))
        
        print('Epoch Number:', i)
        print('Training Loss function value: ', CE(trainTarget, a_output))
        print("Training Accuracy", acc)
        print("Validation Accuracy", acc_v)
        print("Test Accuracy", acc_t)
        
    return W_o, b_o, W_h, b_h, acc_train, loss_train, acc_valid, loss_valid, acc_test, loss_test
        
        
    
    
def test_function():
    #CONSTANTS
    cLearningRate = 1E-5
    cGamma = 0.9
    cEpochs = 200
    cK = 100
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = reshape_data_tensor(trainData)
    #print(trainData.shape)
    validData = reshape_data_tensor(validData)
    testData = reshape_data_tensor(testData)
    trainTarget2 = reshape_target_tensor(trainTarget)
    validTarget2 = reshape_target_tensor(validTarget)
    testTarget2 = reshape_target_tensor(testTarget)
    newTrain, newValid, newTest = convertOneHot(trainTarget, validTarget, testTarget)
    
    #print(trainData.shape[1], ck, trainData.shape[0])
    W_h_init = init_weight_vector(trainData.shape[1], cK)
    v_h_init = np.full((trainData.shape[1], cK), 1E-5)
    b_h_init = np.full((1,cK), 1E-5)
    W_o_init = init_weight_vector(cK,10)
    v_o_init = np.full((cK, 10), 1E-5)
    b_o_init = np.full((1,10), 1E-5)
    
    #print(W_o_init)
    #print(W_h_init)

    W_o, b_o, W_h, b_h, acc_train, loss_train, acc_valid, loss_valid, acc_test, loss_test = learning(W_o_init, v_o_init, b_o_init, W_h_init, v_h_init, b_h_init, cEpochs, cGamma, cLearningRate, trainData, newTrain, validData, newValid, testData, newTest )
    iterations = range(cEpochs)
    '''
    plt.plot(iterations, loss_train, label = 'Training Loss')
    plt.plot(iterations, loss_valid, label = 'Validation Loss')
    plt.plot(iterations, loss_test, label = 'Testing Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs Epoch Using a hidden unit size of 100')
    plt.legend(loc='best')
    plt.savefig('1_4_loss_vs_epoch_hidden100.png')
    plt.show()
    '''
    '''
    plt.plot(iterations, acc_train, label = 'Training Accuracy')
    plt.plot(iterations, acc_valid, label = 'Validation Accuracy')
    plt.plot(iterations, acc_test, label = 'Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs Epoch Using a hidden unit size of 2000')
    plt.legend(loc='best')
    plt.savefig('1_4_accuracy_vs_epoch_hidden2000.png')
    plt.show()
    '''
    '''
    np.savetxt("part1_train_loss_g9_200_2000.txt", loss_train, fmt='%f')
    np.savetxt("part1_valid_loss_g9_200_2000.txt", loss_valid, fmt='%f')
    np.savetxt("part1_test_loss_g9_200_2000.txt", loss_test, fmt='%f')
    
    np.savetxt("part1_train_acc_g9_200_2000.txt", acc_train, fmt='%f')
    np.savetxt("part1_valid_acc_g9_200_2000.txt", acc_valid, fmt='%f')
    np.savetxt("part1_test_acc_g9_200_2000.txt", acc_test, fmt='%f')
    
    #pdb.set_trace()
    plt.plot(iterations, loss_train, label = 'Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('training_loss_vs_epoch_9_200_2000.png')
    plt.show()
    
    plt.plot(iterations, acc_train, label = 'Training Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('training_acc_vs_epoch_9_200_2000.png')
    plt.show()
    
    plt.plot(iterations, loss_valid, label = 'Valid Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('valid_loss_vs_epoch_9_200_2000.png')
    plt.show()
    
    plt.plot(iterations, acc_valid, label = 'Valid Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('valid_acc_vs_epoch_9_200_2000.png')
    plt.show()
    
    plt.plot(iterations, loss_test, label = 'Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('test_loss_vs_epoch_9_200_2000.png')
    plt.show()
    
    plt.plot(iterations, acc_test, label = 'Test Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('test_acc_vs_epoch_9_200_2000.png')
    plt.show()
    '''
    return 0

def conv_neural_network():
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData() #data already comes out normalized
    newTrain, newValid, newTest = convertOneHot(trainTarget, validTarget, testTarget)
    
    trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
    validData = validData.reshape(validData.shape[0], 28, 28, 1)
    testData = testData.reshape(testData.shape[0],28, 28, 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu, strides=(1,1), input_shape=(28,28,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(784, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainData, newTrain, batch_size=32, epochs=50)
    
    test_loss, test_acc = model.evaluate(testData, newTest)
    val_loss, val_acc = model.evaluate(validData, newValid)
    train_loss, train_acc = model.evaluate(trainData, newTrain)
    print(train_loss, train_acc)
    print(val_loss, val_acc)
    print(test_loss, test_acc)
    
    '''
    iterations = range(50)
    
    plt.plot(iterations, train_loss, label = 'Training Loss')
    plt.plot(iterations, val_loss, label = 'Validation Loss')
    plt.plot(iterations, test_loss, label = 'Testing Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss vs Epoch Using SGD')
    plt.legend(loc='best')
    plt.savefig('2_2_loss_vs_epoch_SGD.png')
    plt.show()

    plt.plot(iterations, train_acc, label = 'Training Accuracy')
    plt.plot(iterations, val_acc, label = 'Validation Accuracy')
    plt.plot(iterations, test_acc, label = 'Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy vs Epoch Using SGD')
    plt.legend(loc='best')
    plt.savefig('2_2_accuracy_vs_epoch_SGD.png')
    plt.show()
    '''
    return 0
      
    
if __name__ == '__main__':
    #test_function()
    conv_neural_network()

    
    
    


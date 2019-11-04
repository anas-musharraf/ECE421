import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pdb

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def reshape_data_tensor(tensor):
    arg1,arg2,arg3 = tensor.shape
    new_tensor = tensor.reshape((arg1, (arg2*arg3)))
    return new_tensor
    
def MSE(W, b, x, y, reg):
    # Your implementation here
    constant = y.shape[0]
    error = np.matmul(x,W)
    error_2 = error + b - y
    term1 = np.sum(error_2*error_2,axis=0)
    term2 = (reg/2)*np.sum(W*W)
    return (1/constant)*term1 + term2


def gradMSE(W, b, x, y, reg):
    constant = y.shape[0]
    error = np.matmul(x,W)
    error_2 = error + b - y
    term1 = (error_2)*x
    term1 = np.sum(term1,axis=0).reshape((W.shape[0],1))
    grad_wrt_w = (2/constant)*term1 + reg*W #Makes sense in terms of matrix dimensions but not in terms of formula
    grad_wrt_bias = (2/constant)*np.sum(np.matmul(x,W)+b-y,axis=0)
    return grad_wrt_w, grad_wrt_bias


def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, validation_data, validationTarget, test_data, testTarget, lossType="MSE"):
    # Your implementation here
    current_weight = W
    current_bias = b
    training_loss = np.array([])
    training_accuracy = np.array([])
    test_loss = np.array([])
    test_accuracy = np.array([])
    validation_loss = np.array([])
    validation_accuracy = np.array([])
    init_train_acc = calculated_accuracy(current_weight, current_bias, x,y, lossType)
    init_valid_acc = calculated_accuracy(current_weight, current_bias, validation_data, validationTarget, lossType)
    init_test_acc = calculated_accuracy(current_weight, current_bias, test_data, testTarget, lossType)
    training_accuracy = np.append(training_accuracy,init_train_acc)
    validation_accuracy = np.append(validation_accuracy, init_valid_acc)
    test_accuracy = np.append(validation_accuracy,init_test_acc)    #grad_weight, grad_bias = gradMSE(W,b,x,y,reg)
    for i in range(epochs):
        if lossType == "MSE":
            grad_weight, grad_bias = gradMSE(current_weight,current_bias,x,y,reg)
            training_loss =np.append(training_loss,MSE(current_weight, current_bias,x,y,reg))
            validation_loss =np.append(validation_loss,MSE(current_weight, current_bias, validation_data, validationTarget,reg))
            test_loss =np.append(test_loss,MSE(current_weight,current_bias, test_data, testTarget, reg))
        elif lossType=="CE":
            grad_weight, grad_bias = gradCE(current_weight,current_bias,x,y,reg)
            training_loss =np.append(training_loss,crossEntropyLoss(current_weight, current_bias,x,y,reg))
            validation_loss =np.append(validation_loss,crossEntropyLoss(current_weight, current_bias, validation_data, validationTarget,reg))
            test_loss =np.append(test_loss,crossEntropyLoss(current_weight,current_bias, test_data, testTarget, reg))
        
        updated_weight = current_weight - alpha*grad_weight
        updated_bias = current_bias - alpha*grad_bias
        
        updated_train_acc = calculated_accuracy(updated_weight, updated_bias, x,y, lossType)
        updated_valid_acc = calculated_accuracy(updated_weight,updated_bias, validation_data, validationTarget, lossType)
        updated_test_acc = calculated_accuracy(updated_weight, updated_bias, test_data, testTarget, lossType)

        training_accuracy = np.append(training_accuracy,updated_train_acc)
        validation_accuracy = np.append(validation_accuracy, updated_valid_acc)
        test_accuracy = np.append(validation_accuracy,updated_test_acc)
        
        test = np.sum(grad_weight*grad_weight,axis=0)
        if alpha*np.sum(grad_weight*grad_weight,axis=0) < error_tol:
            return updated_weight, updated_bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy

        current_bias,current_weight = updated_bias, updated_weight
    return updated_weight, updated_bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy

def calculated_accuracy(W,b,x,y, losstype="MSE"):
    if losstype=="CE":
        updated_prediction = sigmoid(W,x,b)
    else:
        updated_prediction = np.matmul(x,W) + b

    updated_prediction = np.round(updated_prediction)
    accuracy = np.sum(updated_prediction==y)/np.shape(x)[0]
    return accuracy
    

def sigmoid(W,x,b):
    term1 = np.matmul(x,W) + b
    top = np.exp(term1)
    return top/(1 + top)
 
    
def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    a = y * np.log(sigmoid(W,x,b))
    b = (1 - y) * np.log(1 - sigmoid(W,x,b))
    N = y.shape[0]
    term1 = np.sum(-a - b, axis=0)/N
    term2 = (reg/2)*np.sum(W*W)
    return term1 + term2

def gradCE(W, b, x, y, reg):
    # Your implementation here
    N = y.shape[0]
    y_hat = sigmoid(W,x,b)
    y_hat_min_y = y_hat-y
    part1Done = y_hat_min_y*x
    sol = (1/N)*np.sum(part1Done,axis=0).reshape([W.shape[0],1]) + reg*W
    part2 = (1/N)*np.sum(y_hat_min_y,axis=0)
    return sol, part2
    
def tf_calc_accuracy(predictions, labels):
    predictions = np.round(predictions)
    accuracy = np.sum(predictions==labels)/np.shape(labels)[0]
    return accuracy


def optWeights(x, y):
    d = np.shape(x)[0]
    one_vec = np.expand_dims(np.ones(d),1)
    x_aug = np.concatenate((x,one_vec),axis = 1)
    A = np.linalg.inv(np.matmul(np.transpose(x_aug),x_aug))
    B = np.matmul(np.transpose(x_aug),y)
    W_aug = np.matmul(A,B)
    W = W_aug[:len(W_aug)-1]
    b = W_aug[len(W_aug)-1]
    return np.transpose(W),b
    
    
def buildGraph(loss="MSE", betaOne="None", betaTwo="None", epsilon="None"):
    lambd = tf.constant([0.0])
    #Initialize weight and bias tensors
    W = tf.Variable(tf.truncated_normal(shape=(784,1),mean=0.0,stddev=0.5,dtype=tf.float32,seed=None,name=None))
    b = tf.Variable(tf.zeros(1))
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    reg = tf.constant([0.0])
    
    tf.set_random_seed(421)
    
    if loss == "MSE":
        y_Pred = tf.matmul(x,W) + b
        loss = tf.losses.mean_squared_error(y,y_Pred)
        #reg = tf.nn.l2_loss(W)
        loss = loss + lambd * reg
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        y_Pred = tf.round(y_Pred) 
        
        
    elif loss == "CE":
        sigmoid_term = tf.matmul(x,W) + b
        y_Pred = tf.sigmoid(sigmoid_term)
        loss = tf.losses.sigmoid_cross_entropy(y, sigmoid_term)
        #ce = tf.losses.sigmoid_cross_entropy( multi_class_labels = tf.reshape(y, [tf.shape(y)[0],1]),logits = pred_y)

        #reg = tf.nn.l2_loss(W)
        loss = loss + lambd * reg
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        y_Pred = tf.round(y_Pred)
        
    
    trainingLoss = []
    trainingAcc = []
    validationLoss = []  
    validationAcc = []
    testingLoss = []
    testingAcc = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
    
        session.run(init)
        trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
        n_batches = int(3500/batch_size)
        for i in range(epochs):
            #Shuffle here

            trainData_shaped = reshape_data_tensor(trainData)
            random_order = np.random.permutation(trainData_shaped.shape[0])
            trainData_random = trainData_shaped[random_order]
            trainTarget_random = trainTarget[random_order]


            validData_shaped = reshape_data_tensor(validData)
            random_order = np.random.permutation(validData_shaped.shape[0])
            validData_random = validData_shaped[random_order]
            validTarget_random = validTarget[random_order]

            
            testData_shaped = reshape_data_tensor(testData)
            random_order = np.random.permutation(testData_shaped.shape[0])
            testData_random = testData_shaped[random_order]
            testTarget_random = testTarget[random_order]


            training_loss_in_batch = 0
            valid_loss_in_batch = 0
            testing_loss_in_batch = 0
            
            trainingPredictions = []
            validPredictions = []
            testingPredictions = []

            trainingCorrect_points = 0
            validCorrect_points = 0
            testCorrect_points = 0
            for i in range(n_batches):
                #print(i)
                #pick first 500 from training data
                feed_x = trainData_random[i*batch_size:((i+1)*batch_size)]
                feed_y = trainTarget_random[i*batch_size:((i+1)*batch_size)]

                validFeed_x = validData_random[i*batch_size:((i+1)*batch_size)]
                validFeed_y = validTarget_random[i*batch_size:((i+1)*batch_size)]

                testingFeed_x = testData_random[i*batch_size:((i+1)*batch_size)]
                testingFeed_y = testTarget_random[i*batch_size:((i+1)*batch_size)]


                session.run(optimizer, feed_dict={x:feed_x, y:feed_y})
                training_loss_in_batch += session.run(loss, feed_dict={x: feed_x, y:feed_y})[0]
                valid_loss_in_batch += session.run(loss, feed_dict={x: validFeed_x, y:validFeed_y})[0]
                testing_loss_in_batch += session.run(loss, feed_dict={x: testingFeed_x, y:testingFeed_y})[0]
                

                trainingPredictions = session.run(y_Pred,feed_dict={x: feed_x, y:feed_y})
                trainingPredictions = np.asarray(trainingPredictions)
                validPredictions = session.run(y_Pred,feed_dict={x: validFeed_x, y:validFeed_y})
                validPredictions = np.asarray(validPredictions)
                testingPredictions = session.run(y_Pred,feed_dict={x: testingFeed_x, y:testingFeed_y})
                testingPredictions = np.asarray(testingPredictions)
                
                trainingCorrect_points += np.sum(trainingPredictions==feed_y)
                validCorrect_points += np.sum(validPredictions==validFeed_y)
                testCorrect_points += np.sum(testingPredictions==testingFeed_y)
                
                #print(trainingCorrect_points)
                
            
            trainingPredictions = np.asarray(trainingPredictions)
            validPredictions = np.asarray(validPredictions)
            testingPredictions = np.asarray(testingPredictions)
            
            
            trainingAcc.append(trainingCorrect_points/trainTarget_random.shape[0])
            trainingLoss.append(training_loss_in_batch/n_batches)
            validationAcc.append(validCorrect_points/validTarget_random.shape[0])
            validationLoss.append(valid_loss_in_batch/n_batches)
            testingAcc.append(testCorrect_points/testTarget_random.shape[0])
            testingLoss.append(testing_loss_in_batch/n_batches)
            #print(trainingAcc)

    return W,b,y_Pred,y,loss,optimizer, trainingLoss, trainingAcc,validationLoss,validationAcc, testingAcc,testingLoss

batch_size = 1750
epochs = 700

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
mu, sigma = 0, 0.5
#betaOne="None", betaTwo="None", epsilon="None"
W, b, y_Pred, y, loss, optimizer, trainingLoss, trainingAcc, validationLoss,validationAcc, testingAcc,testingLoss = buildGraph("CE")

'''
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
x = reshape_data_tensor(trainData)
mu, sigma = 0, 0.5
W = np.random.normal(mu, sigma, (x.shape[1],1))/1000
error_tol = 1e-07                                 
'''

#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.005, 5000, 0, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)
#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.001, 5000, 0, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)
#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.0001, 5000, 0, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)

#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.005, 5000, 0.001, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)
#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.005, 5000, 0.1, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)
#weight, bias, training_loss, validation_loss, test_loss, test_accuracy, validation_accuracy, training_accuracy = grad_descent(W,1,reshape_data_tensor(trainData), trainTarget, 0.005, 5000, 0.5, error_tol, reshape_data_tensor(validData), validTarget, reshape_data_tensor(testData), testTarget)


iterations = range(epochs)
plt.plot(iterations, trainingLoss, label = 'Training Loss')
plt.plot(iterations, validationLoss, label = 'Validation Loss')
plt.plot(iterations, testingLoss, label = 'Testing Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss vs Epoch Using ADAM Optimizer with CE batch size = 1750')
plt.legend(loc='best')
plt.savefig('3_5_loss_vs_epoch_batch1750.png')
plt.show()


iterations = range(epochs)
plt.plot(iterations, trainingAcc, label = 'Training Accuracy')
plt.plot(iterations, validationAcc, label = 'Validation Accuracy')
plt.plot(iterations, testingAcc, label = 'Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy vs Epoch Using ADAM Optimizer with CE batch size = 1750')
plt.legend(loc='best')
plt.savefig('3_5_accuracy_vs_epoch_batch1750.png')
plt.show()








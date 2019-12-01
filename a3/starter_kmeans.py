import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
'''
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]
'''

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    x_update = tf.expand_dims(X,1) #turn matrix into Nx1xD
    mu_update = tf.expand_dims(MU,0) #turn matrix into 1xkxD
    subtract = x_update - mu_update #subtract 1xD  and KxD matrices
    squared_val = tf.square(subtract) #square the distance between each dimension of the vectors
    return tf.reduce_sum(squared_val, 2) #summing over the D dimensions

def buildGraph(dim=2, k=3):
    
    #Inputs
    x = tf.compat.v1.placeholder(tf.float32, [None, dim])
    mu = tf.Variable(tf.random.normal(shape=[k, dim]))
    
    distances = distanceFunc(x,mu)
    
    loss = tf.reduce_sum(tf.math.reduce_min(distances, axis=1))
    prediction = tf.argmin(distances,1)
    
    train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)

    return x, mu, distances, loss, prediction, train
    
#KList = [1, 2, 3, 4, 5]

KList = [3]

best_MUs = []
KTrainLosses = []
KValLosses = []
KTrainCounts = []
KTrainPercents = []
KValCounts = []
KValPercents = []

for K in KList:
    
    print('Running K means clustering on 2D dataset with K = {} ...\n'.format(K))
    
    # build the graph
    MU, X, distances, assignments, loss, train = buildGraph(K, 2)
    
    init = tf.compat.v1.global_variables_initializer()
    
    losses = []
    
    with tf.compat.v1.Session() as sess:
        
        sess.run(init)
        for epoch in range(200):
            
            sess.run([train], feed_dict={X: data})
            lossUpdate = sess.run([loss], feed_dict={X: data})
            losses.append(lossUpdate)
            
            if (epoch + 1) % 10 == 0:
                print('Epoch: {}, training loss: {}'.format(epoch+1,lossUpdate[0]))
        
        # compute best assignments as numpy array
        train_assignments = sess.run([assignments], feed_dict={X: data})[0]
        val_assignments = sess.run([assignments], feed_dict={X: val_data})[0]
        val_loss = sess.run([loss], feed_dict={X: val_data})[0]
        best_MU = MU.eval()
        
    print('\nFinished updating.\nValidation loss after {} updates: {:.5f}\n'.format(epoch+1, val_loss))
    
    unique, counts = np.unique(train_assignments, return_counts=True)
    percentages = counts/np.sum(counts)
    train_counts = dict(zip(unique, counts))
    train_percentages = dict(zip(unique, percentages))
    
    unique, counts = np.unique(val_assignments, return_counts=True)
    percentages = counts/np.sum(counts)
    val_counts = dict(zip(unique, counts))
    val_percentages = dict(zip(unique, percentages))
    
    for i in range(K):
        print('Percentage of training points belonging to cluster {}: {:.2f}%'.format(i, 100*train_percentages[i]))
    print('\n')
    for i in range(K):
        print('Percentage of validation points belonging to cluster {}: {:.2f}%'.format(i, 100*val_percentages[i]))
    print('\n')
    
    best_MUs.append(best_MU)
    KTrainLosses.append(losses)
    KValLosses.append(val_loss)
    KTrainCounts.append(train_counts)
    KTrainPercents.append(train_percentages)
    KValCounts.append(val_counts)
    KValPercents.append(val_percentages)
    
    plt.figure()
    plt.scatter(data[:,0], data[:,1], s=1, c=train_assignments)
    plt.scatter(best_MU[:,0], best_MU[:,1], marker='x', s=25, c='black')
    plt.title('K means clustering on training data with K = {}'.format(K))
    plt.show()
    
    plt.figure()
    plt.scatter(val_data[:,0], val_data[:,1], s=1, c=val_assignments)
    plt.scatter(best_MU[:,0], best_MU[:,1], marker='x', s=25, c='black')
    plt.title('K means clustering on validation data with K = {}'.format(K))
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Number of updates')
    plt.ylabel('Training loss')
    plt.title('K means Training loss vs. number of updates for K = {}'.format(K))
    plt.show()
    
plt.figure()
plt.plot(KList, KValLosses)
plt.xlabel('K')
plt.ylabel('Validation loss')
plt.title('K means validation loss vs. K')
plt.show()



### 100d dataset

data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
#if is_valid:
valid_batch = int(num_pts / 3.0)
np.random.seed(45689)
rnd_idx = np.arange(num_pts)
np.random.shuffle(rnd_idx)
val_data = data[rnd_idx[:valid_batch]]
data = data[rnd_idx[valid_batch:]]


KList_100D = [5, 10, 15, 20, 30]

best_MUs_100D = []
KTrainLosses_100D = []
KValLosses_100D = []
KTrainCounts_100D = []
KTrainPercents_100D = []
KValCounts_100D = []
KValPercents_100D = []

for K in KList_100D:
    
    print('Running K means clustering on 100D dataset with K = {} ...\n'.format(K))
    
    # build the graph
    MU, X, distances, assignments, loss, train = buildGraph(K, 100)
    
    init = tf.compat.v1.global_variables_initializer()
    
    losses = []
    
    with tf.compat.v1.Session() as sess:
        
        sess.run(init)
        for epoch in range(200):
            
            sess.run([train], feed_dict={X: data})
            lossUpdate = sess.run([loss], feed_dict={X: data})
            losses.append(lossUpdate)
            
            if (epoch + 1) % 10 == 0:
                print('Epoch: {}, training loss: {}'.format(epoch+1,lossUpdate[0]))
        
        # compute best assignments as numpy array
        train_assignments = sess.run([assignments], feed_dict={X: data})[0]
        val_assignments = sess.run([assignments], feed_dict={X: val_data})[0]
        val_loss = sess.run([loss], feed_dict={X: val_data})[0]
        best_MU = MU.eval()
        
    print('\nFinished updating.\nValidation loss after {} updates: {:.5f}\n'.format(epoch+1, val_loss))
    
    unique, counts = np.unique(train_assignments, return_counts=True)
    percentages = counts/np.sum(counts)
    train_counts = dict(zip(unique, counts))
    train_percentages = dict(zip(unique, percentages))
    for i in range(K):
        if i not in train_percentages:
            train_counts[i] = 0
            train_percentages[i] = 0
    
    unique, counts = np.unique(val_assignments, return_counts=True)
    percentages = counts/np.sum(counts)
    val_counts = dict(zip(unique, counts))
    val_percentages = dict(zip(unique, percentages))
    for i in range(K):
        if i not in val_percentages:
            val_counts[i] = 0
            val_percentages[i] = 0
    
    for i in range(K):
        print('Percentage of training points belonging to cluster {}: {:.2f}%'.format(i, 100*train_percentages[i]))
    print('\n')
    for i in range(K):
        print('Percentage of validation points belonging to cluster {}: {:.2f}%'.format(i, 100*val_percentages[i]))
    print('\n')
    
    best_MUs_100D.append(best_MU)
    KTrainLosses_100D.append(losses)
    KValLosses_100D.append(val_loss)
    KTrainCounts_100D.append(train_counts)
    KTrainPercents_100D.append(train_percentages)
    KValCounts_100D.append(val_counts)
    KValPercents_100D.append(val_percentages)
    
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Number of updates')
    plt.ylabel('Training loss')
    plt.title('100D K means training loss vs. number of updates for K = {}'.format(K))
    plt.show()
    
plt.figure()
plt.plot(KList_100D, KValLosses_100D)
plt.xlabel('K')
plt.ylabel('Validation loss')
plt.title('100D K means validation loss vs. K')
plt.show()
    
    
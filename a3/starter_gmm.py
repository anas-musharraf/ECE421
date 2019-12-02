import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Loading data
#data = np.load('data100D.npy')
#data = np.load('data2D.npy')
#[num_pts, dim] = np.shape(data)

# For Validation set


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    X_expanded = tf.expand_dims(X, 0)
    MU_expanded = tf.expand_dims(MU, 1)
    pair_dist = tf.transpose(tf.reduce_sum(tf.square(tf.subtract(X_expanded, MU_expanded)), 2))
    return pair_dist

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    D = X.shape[1]
    dist = distanceFunc(X, mu)
    sig = tf.transpose(sigma)
    var = tf.square(sig)
    return -(D/2)*tf.math.log(2*np.pi*var) - dist/(2*var)

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    return tf.transpose(log_pi) + log_PDF - hlp.reduce_logsumexp(tf.transpose(log_pi) + log_PDF, keep_dims=True)


def buildGraph(K, D):
    
    tf.compat.v1.set_random_seed(421)
    MU = tf.Variable(tf.random.normal(shape=[K, D]))
    psi = tf.Variable(tf.random.normal(shape=[K, 1]))
    phi = tf.Variable(tf.random.normal(shape=[K, 1]))
    X = tf.compat.v1.placeholder(tf.float32, [None, D])
    
    sigma = tf.sqrt(tf.exp(phi))
    log_pi = hlp.logsoftmax(psi)
    pi = tf.math.exp(log_pi)
    
    logPDF = log_GaussPDF(X, MU, sigma)
    
    post = log_posterior(logPDF, log_pi)
    assignments = tf.math.argmax(post, axis=1)
    
    loss = -tf.reduce_sum(hlp.reduce_logsumexp(tf.transpose(log_pi) +
                                               logPDF, keep_dims=True))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1, beta1=0.9,
                                             beta2=0.99, epsilon=1e-5)
    
    train = optimizer.minimize(loss)
    
    
    return MU, sigma, pi, X, assignments, loss, train


tf.compat.v1.disable_eager_execution()
'''  
KList = [1, 2, 3, 4, 5]

best_MUs = []
best_sigmas = []
best_pis = []
KTrainLosses = []
KValLosses = []
KTrainCounts = []
KTrainPercents = []
KValCounts = []
KValPercents = []

for K in KList:
    
    print('Running GMM clustering on 2D dataset with K = {} ...\n'.format(K))
    
    # build the graph
    MU, sigma, pi, X, assignments, loss, train = buildGraph(K, 2)
    
    init = tf.compat.v1.global_variables_initializer()
    
    losses = []
    
    with tf.compat.v1.Session() as sess:
        
        sess.run(init)
        for epoch in range(500):
            
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
        best_sigma = sigma.eval()
        best_pi = pi.eval()
        
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
    
    best_MUs.append(best_MU)
    best_sigmas.append(best_sigma)
    best_pis.append(best_pi)
    KTrainLosses.append(losses)
    KValLosses.append(val_loss)
    KTrainCounts.append(train_counts)
    KTrainPercents.append(train_percentages)
    KValCounts.append(val_counts)
    KValPercents.append(val_percentages)
    
    plt.figure()
    plt.scatter(data[:,0], data[:,1], s=1, c=train_assignments)
    plt.scatter(best_MU[:,0], best_MU[:,1], marker='x', s=25, c='black')
    plt.title('GMM clustering on data with K = {}'.format(K))
    plt.savefig('2_2_2_scatter_with_K={}.png'.format(K))
    plt.show()
    
    
    plt.figure()
    plt.scatter(val_data[:,0], val_data[:,1], s=1, c=val_assignments)
    plt.scatter(best_MU[:,0], best_MU[:,1], marker='x', s=25, c='black')
    plt.title('GMM clustering on validation data with K = {}'.format(K))
    plt.show()
    
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Number of updates')
    plt.ylabel('Training loss')
    plt.title('GMM Training loss vs. number of updates for K = {}'.format(K))
    plt.savefig('2_2_1_loss_vs_epoch_with_K={}.png'.format(K))
    plt.show()
    
    plt.figure()
    plt.plot(KList, KValLosses)
    plt.xlabel('K')
    plt.ylabel('Validation loss')
    plt.title('GMM Validation loss vs. K')
    plt.show()
    
'''
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


KList_100D = [20, 30]

best_MUs_100D = []
best_sigmas_100D = []
best_pis_100D = []
KTrainLosses_100D = []
KValLosses_100D = []
KTrainCounts_100D = []
KTrainPercents_100D = []
KValCounts_100D = []
KValPercents_100D = []

for K in KList_100D:
    
    print('Running GMM clustering on 100D dataset with K = {} ...\n'.format(K))
    
    # build the graph
    MU, sigma, pi, X, assignments, loss, train = buildGraph(K, 100)
    
    init = tf.compat.v1.global_variables_initializer()
    
    losses = []
    
    with tf.compat.v1.Session() as sess:
        
        sess.run(init)
        for epoch in range(500):
            
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
        best_sigma = sigma.eval()
        best_pi = pi.eval()
        
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
    best_sigmas_100D.append(best_sigma)
    best_pis_100D.append(best_pi)
    KTrainLosses_100D.append(losses)
    KValLosses_100D.append(val_loss)
    KTrainCounts_100D.append(train_counts)
    KTrainPercents_100D.append(train_percentages)
    KValCounts_100D.append(val_counts)
    KValPercents_100D.append(val_percentages)
    '''
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Number of updates')
    plt.ylabel('Training loss')
    plt.title('100D GMM Training loss vs. number of updates for K = {}'.format(K))
    plt.show()
    
    plt.figure()
    plt.plot(KList_100D, KValLosses_100D)
    plt.xlabel('K')
    plt.ylabel('Validation loss')
    plt.title('100D GMM Validation loss vs. K')
    plt.show()
    '''
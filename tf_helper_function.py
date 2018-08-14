import h5py
import numpy as np 
import math


def load_datasets():
	train_datasets = h5py.File('train_signs.h5',"r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File('test_signs.h5',"r")
	test_set_x_orig =  np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def create_placeholders(n_h0,n_w0,n_c0,n_y):
	"""
    n_h0 - height of input image
    n_w0 - widht of an input image
    n_c0 - number of channel of the input
    n_y - number of classes
	"""

	x = tf.placeholder(tf.float32,[None, n_h0,n_w0,n_c0])
	y = tf.placeholder(tf.float32, [None, n_y])

	return x, y



def initialize_parameters():
	tf.set_random_seed(1)
	w1 = tf.get_variable("w1", [4,4,3,8], initialization = tf.contrib.layers.xavier_initializer(seed = 0))
	w2 = tf.get_variable("w2",[2,2,8,16], initialization = tf.contrib.layers.xavier_initializer(seed = 0))

	parameters = {"w1":w1,"w2":w2}

	return parameters

def forward_propagation(x, parameters):
	
	w1 = parameters['w1']
	w2 = parameters['w2']
	z1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME')
	a1 = tf.nn.relu(z1)
	p1 = tf.nn.max_pool(a1,ksize = [1,8,8,1], strides = [1,8,8,1], padding='SAME')
	z2 = tf.nn.conv2d(p1,w2,strides= [1,1,1,1], padding = 'SAME')
    a2 = tf.nn.relu(z2)
    p2 = tf.nn.max_pool(a2,ksize = [1,4,4,1],strides = [1,4,4,1	], padding = 'SAME')
    p = tf.contrib.layers.flatten(p2)
    z3 = tf.contrib.layers.fully_connected(p,6,activation_fn = None)

    return z3

def compute_cost(z3,y):
	cost = tf.reduce.mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3,labels=y))
	return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
    	mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    	mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
    	mini_batch = (mini_batch_X, mini_batch_Y)
    	mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    
    return mini_batches
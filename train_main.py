
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_helper_function import *



def model(x_train,y_train,x_test,y_test, learning_rate = 0.009, num_epochs=100, minibatch_sized=64,print_cost = True):

    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


	ops.reset_default_graph()
	tf.set_random_seed(1)
	(m,n_h0,n_w0,n_c0,) = x.train.shape
	n_y = y_train.shape[1]

	costs = []

	x, y  = create_placeholders(n_h0,n_w0,n_c0,n_y)

	parameters = initialize_parameters()

	z3 =  forward_propagation(x, parameters)

	cost = compute_cost(z3,y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()


    with tf.Session() as sess:
    	sess.run(init)
    	for epoch in range(num_epochs):
    		minibatch_cost = 0
    		num_minibatches = int(m/minibatch_size)
    		seed = seed +1
    		minibatches = random_mini_batches(x_train,y_train,minibatch_size, seed)
    		for minibatch in minibatches:
    			(minibatch_x,minibatch_y) = minibatch

    			_,temp_cost = sess.run([optimizer,cost],feed_dict = {x:minibatch_x,y:minibatch_y})
    			minibatch_cost += temp_cost / num_minibatches

    		if print_cost == True and epoch % 5 ==0:
    			print("cost  after epoch %i: %f"%(epoch,minibatch_cost))
    			costs.append(minibatch_cost)

    		if print_cost == True and epoch %1 == 0:
    			costs.append(minibatch_cost)


    	plt.plot(np.squueze(costs))
    	plt.ylabel('cost')
    	plt.xlabel(' iteration  (per tens)')
    	plt.title("learning rate =", str(learning_rate))
    	plt.show()

    	#calculate the correct prediction
    	predict_op = tf.argmax(z3,1)
    	correct_prediction = tf.equal(predict_op,tf.argmax(y,1))

    	accuarcy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    	print(accuracy)

    	#train_accuracy
    	train_accuracy = accuracy.eval({x: x_train, y: y_train})
    	test_accuracy = accuracy.eval({x:x_test, y:y_test})
    	print("train accuracy", train_accuracy)
    	print("test accuarcy", test_accuracy)

    return train_accuracy,test_accuracy, parameters

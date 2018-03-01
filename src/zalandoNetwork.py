
# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

# Import Fashion MNIST
fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)


def model(train, test, learning_rate=0.0005, num_epochs=16, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()

    (n_x, m) = train.images.T.shape
    n_y = train.labels.T.shape[0]

    # Input: flattened 28x28 images
    input = tf.placeholder(tf.float32, [n_x, None])
    X = tf.transpose(input)
    # Reshape to 28x28 
    input_ = tf.reshape(X, shape=[-1, 28, 28, 1])

    K = 8 # Number of filters for CONV1
    #L = 8 # Number of filters for CONV2
    M = 256 # Number of nodes in HL2

    # Weights and biases for CONV1, filtersize = 8x8
    W1 = tf.Variable(tf.truncated_normal([8, 8, 1, K], stddev=0.1))
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    # Weights and biases for CONV2, filtersize = 4x4
    #W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
    #B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

    # Weights and biases for connections between HL1 and HL2
    W4 = tf.Variable(tf.truncated_normal([968, M], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
    # Weights and biases for connections between HL2 and OUTPUT
    W5 = tf.Variable(tf.truncated_normal([M, n_y], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [n_y]))

    # The two convolutional layers:
    stride = 2  # output = 14x14
    Y1 = tf.nn.relu(tf.nn.conv2d(input_, W1, strides=[1, stride, stride, 1], padding='VALID') + B1) #CONV1
    #Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2) #CONV2


    # reshape the output from the last convolution for the fully connected layer
    Y3 = tf.reshape(Y1, shape=[-1,968]) #HL1
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4) #HL2
    Y = tf.matmul(Y4, W5) + B5 # OUTPUT

    labels = tf.placeholder(tf.float32, [n_y, None])
    output = tf.transpose(Y)


    # Compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=tf.transpose(labels)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        costs = []
        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)

            for i in range(num_minibatches):
                minibatch_X, minibatch_Y = train.next_batch(minibatch_size)
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={input: minibatch_X.T, labels: minibatch_Y.T})
                 # Update epoch cost
                epoch_cost += minibatch_cost / num_minibatches
                
            # Print the cost every epoch
            if print_cost == True:
                print("Cost after epoch {epoch_num}: {cost}".format(epoch_num=epoch, cost=epoch_cost))
                costs.append(epoch_cost)
    
        # Calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(output), tf.argmax(labels))
    
        # Calculate accuracy 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({input: train.images.T, labels: train.labels.T}))
        print ("Test Accuracy:", accuracy.eval({input: test.images.T, labels: test.labels.T}))
        
train = fashion_mnist.train
test = fashion_mnist.test

_ = model(train, test)
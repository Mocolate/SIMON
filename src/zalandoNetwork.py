
# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

# Import Fashion MNIST
fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)


class cnn():
    def _init_(self):
        # Input: flattened 28x28 images
        self.input = tf.placeholder(tf.float32, [None, 748])
        # Reshape to 28x28 
        self.input_ = tf.reshape(self.X, shape=[-1, 28, 28, 1])

        K = 4 # Number of filters for CONV1
        L = 8 # Number of filters for CONV2
        M = 128 # Number of nodes in HL2

        # Weights and biases for CONV1, filtersize = 8x8
        W1 = tf.Variable(tf.truncated_normal([8, 8, 1, K], stddev=0.1))
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        # Weights and biases for CONV2, filtersize = 4x4
        W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

        # Weights and biases for connections between HL1 and HL2
        W4 = tf.Variable(tf.truncated_normal([392, M], stddev=0.1))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
        # Weights and biases for connections between HL2 and OUTPUT
        W5 = tf.Variable(tf.truncated_normal([M, 10], stddev=0.1))
        B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

        # The two convolutional layers:
        stride = 2  # output = 14x14
        self.Y1 = tf.nn.relu(tf.nn.conv2d(self.X_, W1, strides=[1, stride, stride, 1], padding='VALID') + B1) #CONV1
        stride = 2  # output = 7x7
        self.Y2 = tf.nn.relu(tf.nn.conv2d(self.Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2) #CONV2
       
        # reshape the output from the last convolution for the fully connected layer
        self.Y3 = tf.reshape(self.Y2, shape=[-1, 392]) #HL1
        self.Y4 = tf.nn.relu(tf.matmul(self.YY, W4) + B4) #HL2
        self.Y = tf.matmul(self.YY4, W5) + B5 # OUTPUT

        self.label = tf.placeholder(tf.float32, [None])

        # Compute cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Y, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

        # Calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(Y), tf.argmax(labels))
        
        # Calculate accuracy 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        

#Init:
network = cnn()
init = tf.global_variables_initializer
epochs = 15

with tf.Session() as sess:
    sess.run(init)

    finalaccuracy = 0

    for epoch in range(epochs):
        
        _, epoch_cost, accuracy = sess.run([optimizer, cost, accuracy], feed_dict={X: tf.fashion_mnist.train.images, Y: tf.fashion_mnist.train.labels})
                
        # Print the cost every epoch
        print("Cost after epoch {epoch_num}: {cost}".format(epoch_num=epoch, cost=epoch_cost))
        trainaccuracy = accuracy

    testaccurracy = sess.run(accuracy, feed_dict={X: tf.fashion_mnist.test.images, Y: tf.fashion_mnist.test.labels})

    print("Training accuracy: " + trainaccuracy)
    print("Testing accuracy: " + testaccurracy)
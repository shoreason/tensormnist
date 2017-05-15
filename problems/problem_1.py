import sys, os
sys.path.append(sys.path.append(os.path.dirname(__file__) + "/.."))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_neural_network import mnist_neural_network


def variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)


def create_model(x, input_size, output_size):
    hidden_layer_size = 800

    # Hidden layer 1
    # Insert your code here!
    W1 = variable([input_size, hidden_layer_size], "W1")
    b1 = variable([hidden_layer_size], "b1")
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1, name = "h1")
    
    # Hidden layer 2
    # Insert your code here!
    W2 = variable([hidden_layer_size, hidden_layer_size], "W2")
    b2 = variable([hidden_layer_size], "b2")
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2, name = "h2")

    # Output layer
    W_out = variable([hidden_layer_size, output_size], "W_out")
    b_out = variable([output_size], "b_out")
    y = tf.nn.softmax(tf.matmul(h2, W_out) + b_out, name = "y")

    return y


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
nn = mnist_neural_network.MnistNeuralNetwork(mnist, create_model)
nn.train_and_evaluate()

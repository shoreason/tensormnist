import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Hyperparameters
hidden_layer_size = 800
batch_size = 50
learning_rate = 0.05
iterations = 4000

# Other constants
input_size = 28 * 28 # 784
output_size = 10

# Just a little function to define a TensorFlow variable.  We make our variables'
# initial values slightly positive because we use a ReLU activation function.  This
# reduces the chance of dead neurons.
def variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)

# Inputs
x = tf.placeholder(tf.float32, shape = [None, input_size], name = "x")
y_ = tf.placeholder(tf.float32, shape = [None, output_size], name = "y_")

# Hidden layer
W1 = variable([input_size, hidden_layer_size], "W1")
b1 = variable([hidden_layer_size], "b1")
h1 = tf.nn.relu(tf.matmul(x, W1) + b1, name = "h1") # h = a(W1x + b1)

# Output layer
W_out = variable([hidden_layer_size, output_size], "W_out")
b_out = variable([output_size], "b_out")
y = tf.nn.softmax(tf.matmul(h1, W_out) + b_out, name = "y")

# Cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]), name = "cross_entropy")

# Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy")

# Initialize a TensorFlow session.
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Train the network on the training data.
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
for i in range(iterations):
	batch = mnist.train.next_batch(batch_size)
	if i % 100 == 0:
		train_accuracy = 100 * accuracy.eval(feed_dict = {x: batch[0], y_: batch[1]})
		print("Step {:05d}: batch training accuracy {:.0f}%".format(i, train_accuracy))
	train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

# Evaluate the network on the test data.
print("Accuracy: {:.3f}%".format(100 * accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels})))

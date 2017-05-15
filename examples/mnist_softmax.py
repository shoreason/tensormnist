import tensorflow as tf
# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

hidden_layer_size = 800
learning_rate = 0.1
iteration_count = 3000
batch_size = 100

# images going into input layer (input Layer images)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, hidden_layer_size], stddev=0.1), name="W")
b = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b")

# Hidden layer
h1 = tf.nn.relu(tf.matmul(x, W) + b, name = "h1")
W_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.1), name="W_h1")
b_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b_h1")

h2 = tf.nn.relu(tf.matmul(h1, W_h1) + b_h1, name = "h2")
W_h2 = tf.Variable(tf.truncated_normal([hidden_layer_size, 10], stddev=0.1), name="W_h2")
b_h2 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name="b_h2")

# Actual output
y = tf.nn.softmax(tf.matmul(h2, W_h2) + b_h2, name="y")

# Expected output
y_ = tf.placeholder(tf.float32, [None, 10])

# Cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# create session
sess = tf.InteractiveSession()

# initialize variables
# tf.global_variables_initializer().run()
tf.initialize_all_variables().run()

for _ in range(iteration_count):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

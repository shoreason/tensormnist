import tensorflow as tf
# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

hidden_layer_size = 800
learning_rate = 0.001
iteration_count = 300
batch_size = 100
dropout_keep_prob = 0.5

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


y_ = tf.placeholder(tf.float32, [None, 10])
sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.initialize_all_variables().run()
# sess.run(tf.global_variables_initializer())
for i in range(iteration_count):
  batch = mnist.train.next_batch(batch_size)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_keep_prob})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




#
# # images going into input layer (input Layer images)
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.truncated_normal([784, hidden_layer_size], stddev=0.1), name="W")
# b = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b")
#
# # Hidden layer
# h1 = tf.nn.relu(tf.matmul(x, W) + b, name = "h1")
# W_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.1), name="W_h1")
# b_h1 = tf.Variable(tf.truncated_normal([hidden_layer_size], stddev=0.1), name="b_h1")
#
# h2 = tf.nn.relu(tf.matmul(h1, W_h1) + b_h1, name = "h2")
# W_h2 = tf.Variable(tf.truncated_normal([hidden_layer_size, 10], stddev=0.1), name="W_h2")
# b_h2 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name="b_h2")
#
# # Actual output
# y = tf.nn.softmax(tf.matmul(h2, W_h2) + b_h2, name="y")
#
# # Expected output
# y_ = tf.placeholder(tf.float32, [None, 10])
#
# # Cost function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # create session
# sess = tf.InteractiveSession()
#
# # initialize variables
# # tf.global_variables_initializer().run()
# tf.initialize_all_variables().run()
#
# for _ in range(iteration_count):
#   batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#   print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
#
#
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

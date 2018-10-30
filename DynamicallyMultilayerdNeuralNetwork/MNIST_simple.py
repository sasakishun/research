import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ネットワーク構成
num_units2 = 200
num_units1 = 100

x = tf.placeholder(tf.float32, [None, 784])

w2 = tf.Variable(tf.truncated_normal([784, num_units2]))
b2 = tf.Variable(tf.ones([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(x, w2) + b2)

w1 = tf.Variable(tf.truncated_normal([num_units2, num_units1]))
b1 = tf.Variable(tf.ones([num_units1]))
hidden1 = tf.nn.relu(tf.matmul(hidden2, w1) + b1)

w0 = tf.Variable(tf.ones([num_units1, 10]))
b0 = tf.Variable(tf.ones([10]))
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)


t = tf.placeholder(tf.float32, [None, 10])

loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

i = 0
for _ in range(50000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                            feed_dict={x: mnist.test.images,
                                                       t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20180316)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hidden_unit_num = 500

x = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.truncated_normal(shape=[784, hidden_unit_num], mean=0.0, stddev=1 / 784))
b1 = tf.Variable(tf.constant(0.01, shape=[hidden_unit_num]))
w0 = tf.Variable(tf.truncated_normal(shape=[hidden_unit_num, 10], mean=0.0, stddev=1 / hidden_unit_num))
b0 = tf.Variable(tf.constant(0.01, shape=[10]))

hidden = tf.matmul(x, w1) + b1
f = tf.matmul(tf.nn.relu(hidden), w0) + b0
p = tf.nn.softmax(f)

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(20000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x: mnist.test.images, t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))

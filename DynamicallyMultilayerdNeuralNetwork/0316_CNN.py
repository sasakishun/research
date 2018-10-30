import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20180316)
tf.set_random_seed(20180316)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

num_filters = 16

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters],
                                         stddev=0.1))
h_conv = tf.nn.conv2d(x_image, W_conv,
                      strides=[1, 1, 1, 1], padding='SAME')
h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

h_pool_flat = tf.reshape(h_pool, [-1, 14 * 14 * num_filters])

num_units1 = 14 * 14 * num_filters
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

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
        saver.save(sess, 'mdc_session', global_step=i)

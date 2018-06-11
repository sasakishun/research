import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ネットワーク構成
class phase0:
    x = tf.placeholder(tf.float32, [None, 784])

    w0 = tf.Variable(tf.ones([784, 10]))
    b0 = tf.Variable(tf.ones([10]))
    p = tf.nn.softmax(tf.matmul(x, w0))

    t = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))

    train_step = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())


class phase1:
    num_units1 = 100

    x = tf.placeholder(tf.float32, [None, 784])

    w2_0 = tf.Variable(tf.truncated_normal([784, 10]))

    w1 = tf.Variable(tf.truncated_normal([784, num_units1]))
    b1 = tf.Variable(tf.ones([num_units1]))
    hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w0 = tf.Variable(tf.ones([num_units1, 10]))
    b0 = tf.Variable(tf.ones([10]))
    p = tf.nn.softmax(tf.matmul(hidden1, w0) + tf.matmul(x, w2_0) + b0)

    t = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))

    train_step = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())


class phase2:
    num_units2 = 100
    num_units1 = 100

    x = tf.placeholder(tf.float32, [None, 784])

    w2 = tf.Variable(tf.truncated_normal([784, num_units2]))
    b2 = tf.Variable(tf.ones([num_units2]))
    hidden2 = tf.nn.relu(tf.matmul(x, w2) + b2)

    w3_0 = tf.Variable(tf.truncated_normal([784, 10]))
    # w3_0 = tf.placeholder(tf.float32, [784, 10])
    w2_0 = tf.Variable(tf.truncated_normal([num_units2, 10]))

    w1 = tf.Variable(tf.truncated_normal([num_units2, num_units1]))
    b1 = tf.Variable(tf.ones([num_units1]))
    hidden1 = tf.nn.relu(tf.matmul(hidden2, w1) + b1)

    w0 = tf.Variable(tf.ones([num_units1, 10]))
    b0 = tf.Variable(tf.ones([10]))
    p = tf.nn.softmax(tf.matmul(hidden1, w0) + tf.matmul(hidden2, w2_0) + tf.matmul(x, w3_0) + b0)

    t = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0))) \
           - tf.reduce_sum(w3_0)

    train_step = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())

phase = [phase0(), phase1(), phase2()]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

i = 0
j = 0
for j in range(len(phase)):
    if j == 1:
        phase[j].w2_0 = sess.run(phase[j - 1].w0)
    if j == 2:
        # phase[j].w3_0 = sess.run(phase[j - 1].w2_0)
        phase[j].w2_0 = sess.run(phase[j - 1].w0)
        phase[j].w2 = sess.run(phase[j - 1].w1)
    for _ in range(1000):
        i += 1
        batch_xs, batch_ts = mnist.train.next_batch(100)
        sess.run(phase[j].train_step, feed_dict={phase[j].x: batch_xs, phase[j].t: batch_ts})
        if i % 100 == 0:
            loss_val, acc_val = sess.run([phase[j].loss, phase[j].accuracy],
                                                feed_dict={phase[j].x: mnist.test.images,
                                                           phase[j].t: mnist.test.labels})
            print('Step: %d, Loss: %f, Accuracy: %f'
                  % (i, loss_val, acc_val))
    j += 1

images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(phase[-1].p, feed_dict={phase[-1].x: images, phase[-1].t: labels})

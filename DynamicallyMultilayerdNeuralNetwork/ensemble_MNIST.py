import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ネットワーク構成
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        num_units3 = 200
        num_units2 = 200
        num_units1 = 100
        input_size = 784
        with tf.name_scope('input_layer'):
            x = tf.placeholder(tf.float32, [None, input_size])

        with tf.name_scope('layer3'):
            with tf.name_scope('w3_0'):
                w3_0 = tf.Variable(tf.truncated_normal([input_size, num_units3]))
            with tf.name_scope('b3'):
                b3 = tf.Variable(tf.ones([num_units3]))
            with tf.name_scope('b3_1'):
                b3_1 = tf.Variable(tf.ones([num_units3]))
            with tf.name_scope('hidden3_0'):
                hidden3_0 = tf.nn.relu(tf.matmul(x, w3_0) + b3)
            with tf.name_scope('hidden3_1'):
                hidden3_1 = tf.nn.relu(tf.matmul(x, w3_0) + b3_1)
        with tf.name_scope('layer2'):
            with tf.name_scope('w2_0'):
                w2_0 = tf.Variable(tf.truncated_normal([num_units3, num_units2]))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.ones([num_units2]))
            with tf.name_scope('b2_1'):
                b2_1 = tf.Variable(tf.ones([num_units2]))
            with tf.name_scope('hidden2_0'):
                hidden2_0 = tf.nn.relu(tf.matmul(hidden3_0, w2_0) + b2)
            with tf.name_scope('hidden2_1'):
                hidden2_1 = tf.nn.relu(tf.matmul(hidden3_1, w2_0) + b2_1)
        """with tf.name_scope('layer1'):
            with tf.name_scope('hidden1'):
                hidden1 = (hidden2_0 + hidden2_1)
        """
        with tf.name_scope('output_layer'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(tf.truncated_normal([num_units2, 10]))
            with tf.name_scope('w0_1'):
                w0_1 = tf.Variable(tf.truncated_normal([num_units2, 10]))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.ones([10]))
            with tf.name_scope('P0'):
                p0 = tf.nn.softmax(tf.matmul(hidden2_0, w0) + tf.matmul(hidden2_1, w0_1) + b0)

        with tf.name_scope('optimizer'):
            with tf.name_scope('t'):
                t = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope('loss0'):
                loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0)))
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy0", accuracy)
        tf.summary.scalar("w0", tf.reduce_sum(w0))

        self.x, self.t, self.p = x, t, p0
        self.train_step = train_step0
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist_sl_logs", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer
        # sess = tf.InteractiveSession()
        # sess.run(tf.global_variables_initializer())

nn = layer()
i = 0
j = 0
for _ in range(200000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x: mnist.test.images, nn.t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)

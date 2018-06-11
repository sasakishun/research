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
        num_units2 = 300
        num_units1 = 100
        with tf.name_scope('input_layer'):
            x = tf.placeholder(tf.float32, [None, 784])

        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w2 = tf.Variable(tf.truncated_normal([784, num_units2]))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.ones([num_units2]))
            with tf.name_scope('hidden2_1'):
                hidden2_1 = tf.nn.relu(tf.matmul(x, w2) + b2)
            with tf.name_scope('hidden2_2'):
                hidden2_2 = tf.nn.relu(tf.matmul(x, w2) + b2)

        with tf.name_scope('layer1'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(tf.truncated_normal([num_units2, num_units1]))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.ones([num_units1]))
            with tf.name_scope('hidden1'):
                hidden1 = tf.nn.relu(tf.matmul(hidden2_2, w1) + b1)

        with tf.name_scope('output_layer'):
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.ones([10]))
            with tf.name_scope('W3_0'):
                w3_0 = tf.Variable(tf.truncated_normal([784, 10]))
            with tf.name_scope('P0'):
                p0 = tf.nn.softmax(tf.matmul(x, w3_0) + b0)
            with tf.name_scope('W2_0'):
                w2_0 = tf.Variable(tf.truncated_normal([num_units2, 10]))
            with tf.name_scope('p1'):
                p1 = tf.nn.softmax(tf.matmul(hidden2_1, w2_0) + tf.matmul(x, w3_0) + b0)
            with tf.name_scope('W0'):
                w0 = tf.Variable(tf.ones([num_units1, 10]))
            with tf.name_scope('p2'):
                p2 = tf.nn.softmax(tf.matmul(hidden1, w0) + tf.matmul(hidden2_2, w2_0) + tf.matmul(x, w3_0) + b0)

        with tf.name_scope('optimizer'):
            with tf.name_scope('t'):
                t = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope('loss0'):
                loss0 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0)))
            with tf.name_scope('loss1'):
                loss1 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0))) \
                        + 0.0001 * tf.norm(w3_0) ** 2
            with tf.name_scope('loss2'):
                loss2 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p2, 1e-10, 1.0))) \
                        + 0.0001 * tf.norm(w2_0) ** 2
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer().minimize(loss0)
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer().minimize(loss1)
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer().minimize(loss2)

        with tf.name_scope('evaluator'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = [tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1)),
                                      tf.equal(tf.argmax(p1, 1), tf.argmax(t, 1)),
                                      tf.equal(tf.argmax(p2, 1), tf.argmax(t, 1))]

            with tf.name_scope('accuracy'):
                accuracy = [tf.reduce_mean(tf.cast(correct_prediction[0], tf.float32)),
                            tf.reduce_mean(tf.cast(correct_prediction[1], tf.float32)),
                            tf.reduce_mean(tf.cast(correct_prediction[2], tf.float32))]

        tf.summary.scalar("loss0", loss0)
        tf.summary.scalar("loss1", loss1)
        tf.summary.scalar("loss2", loss2)
        tf.summary.scalar("accuracy0", accuracy[0])
        tf.summary.scalar("accuracy1", accuracy[1])
        tf.summary.scalar("accuracy2", accuracy[2])
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w1", tf.reduce_sum(w1))
        tf.summary.scalar("w2", tf.reduce_sum(w2))
        tf.summary.scalar("w2_0", tf.reduce_sum(w2_0))
        tf.summary.scalar("w3_0", tf.reduce_sum(w3_0))

        self.x, self.t, self.p = x, t, [p0, p1, p2]
        self.train_step = [train_step0, train_step1, train_step2]
        self.loss = [loss0, loss1, loss2]
        self.accuracy = accuracy
        self.w2_0 = w2_0
        self.w3_0 = w3_0

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
epoch = [1000, 1000, 3000]
for j in range(3):

    for _ in range(epoch[j]):
        i += 1
        batch_xs, batch_ts = mnist.train.next_batch(100)
        nn.sess.run(nn.train_step[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss[j], nn.accuracy[j]],
                feed_dict={nn.x: mnist.test.images, nn.t: mnist.test.labels})
            print('Step: %d, Loss: %f, Accuracy: %f'
                  % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)
    j += 1

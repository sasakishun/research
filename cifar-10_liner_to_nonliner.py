import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import load_cifer10
import random
from PIL import Image

np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# import tensorflow.contrib.learn.python.learn.datasets.mnist

# ネットワーク構成
"""
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        input_size = 3072
        num_units2 = 1000
        num_units1 = 1000
        with tf.name_scope('input_layer'):
            x = tf.placeholder(tf.float32, [None, input_size])

        with tf.name_scope('layer2_1'):
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.ones([num_units2]))
            with tf.name_scope('w2_1'):
                w2_1 = tf.Variable(tf.truncated_normal([input_size, num_units2]))
            with tf.name_scope('hidden2_1'):
                hidden2_1 = tf.nn.relu(tf.matmul(x, w2_1) + b2)
        with tf.name_scope('layer2_2'):
            with tf.name_scope('w2_2'):
                w2_2 = tf.Variable(tf.truncated_normal([input_size, num_units2]))
            with tf.name_scope('hidden2_2'):
                hidden2_2 = tf.nn.relu(tf.matmul(x, w2_2) + b2)

        with tf.name_scope('layer1'):
            with tf.name_scope('w1_2'):
                w1_2 = tf.Variable(tf.truncated_normal([num_units2, num_units1]))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.ones([num_units1]))
            with tf.name_scope('hidden1'):
                hidden1 = tf.nn.relu(tf.matmul(hidden2_2, w1_2) + b1)

        with tf.name_scope('output_layer0'):
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.ones([10]))
            with tf.name_scope('W3_0_0'):
                w3_0_0 = tf.Variable(tf.truncated_normal([input_size, 10]))
            with tf.name_scope('P0'):
                p0 = tf.nn.softmax(tf.matmul(x, w3_0_0) + b0)
        with tf.name_scope('output_layer1'):
            with tf.name_scope('W3_0_1'):
                w3_0_1 = tf.Variable(tf.truncated_normal([input_size, 10]))
            with tf.name_scope('W2_0_1'):
                w2_0_1 = tf.Variable(tf.truncated_normal([num_units2, 10]))
            with tf.name_scope('p1'):
                p1 = tf.nn.softmax(tf.matmul(hidden2_1, w2_0_1) + tf.matmul(x, w3_0_1) + b0)
        with tf.name_scope('output_layer2'):
            with tf.name_scope('W3_0_2'):
                w3_0_2 = tf.Variable(tf.truncated_normal([input_size, 10]))
            with tf.name_scope('W2_0_2'):
                w2_0_2 = tf.Variable(tf.truncated_normal([num_units2, 10]))
            with tf.name_scope('W0_2'):
                w0_2 = tf.Variable(tf.ones([num_units1, 10]))
            with tf.name_scope('p2'):
                p2 = tf.nn.softmax(tf.matmul(hidden1, w0_2) + tf.matmul(hidden2_2, w2_0_2) + tf.matmul(x, w3_0_2) + b0)

        with tf.name_scope('optimizer'):
            with tf.name_scope('t'):
                t = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope('loss0'):
                loss0 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0)))
            with tf.name_scope('loss1'):
                loss1 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0))) \
                        + 0.0001 * tf.norm(w3_0_1)**2
            with tf.name_scope('loss2'):
                loss2 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p2, 1e-10, 1.0))) \
                        + 0.0001 * tf.norm(w2_0_2)**2
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
        tf.summary.scalar("w0_2", tf.reduce_sum(w0_2))
        tf.summary.scalar("w1_2", tf.reduce_sum(w1_2))
        tf.summary.scalar("w2_1", tf.reduce_sum(w2_1))
        tf.summary.scalar("w2_2", tf.reduce_sum(w2_2))
        tf.summary.scalar("w2_0_1", tf.reduce_sum(w2_0_1))
        tf.summary.scalar("w2_0_2", tf.reduce_sum(w2_0_2))
        tf.summary.scalar("w3_0_0", tf.reduce_sum(w3_0_0))
        tf.summary.scalar("w3_0_1", tf.reduce_sum(w3_0_1))
        tf.summary.scalar("w3_0_2", tf.reduce_sum(w3_0_2))

        self.x, self.t, self.p = x, t, [p0, p1, p2]
        self.train_step = [train_step0, train_step1, train_step2]
        self.loss = [loss0, loss1, loss2]
        self.accuracy = accuracy
        self.w2_0 = w2_0
        self.w3_0_0 = w3_0_0
        self.w3_0_1 = w3_0_1
        self.w3_0_2 = w3_0_2

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
"""


class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        input_size = 3072
        num_units2 = 1000
        num_units1 = 200
        with tf.name_scope('input_layer'):
            x = tf.placeholder(tf.float32, [None, input_size])

        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w2 = tf.Variable(tf.truncated_normal([input_size, num_units2]))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.ones([num_units2]))
            with tf.name_scope('hidden2'):
                hidden2 = tf.nn.relu(tf.matmul(x, w2) + b2)

        with tf.name_scope('layer1'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(tf.truncated_normal([num_units2, num_units1]))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.ones([num_units1]))
            with tf.name_scope('hidden1'):
                hidden1 = tf.nn.relu(tf.matmul(hidden2, w1) + b1)

        with tf.name_scope('output_layer'):
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.ones([10]))
            with tf.name_scope('W0'):
                w0 = tf.Variable(tf.truncated_normal([num_units1, 10]))
            with tf.name_scope('P0'):
                p0 = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

        with tf.name_scope('optimizer'):
            with tf.name_scope('t'):
                t = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope('loss'):
                loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0)))
                # loss0 = -tf.reduce_sum(t * tf.log(p0))
            with tf.name_scope('train_step0'):
                # train_step = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
                train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss0", loss)
        tf.summary.scalar("accuracy0", accuracy)
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w1", tf.reduce_sum(w1))
        tf.summary.scalar("w2", tf.reduce_sum(w2))

        self.x, self.t, self.p = x, t, p0
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        self.w2 = w2
        self.w1 = w1

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tmp/mnist_sl_logs", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    data, labels_non_onehot, test_data, test_labels_non_onehot, label_names = load_cifer10.load_dataset()
    print(label_names)
    labels = np.mat([[0 for i in range(10)] for k in range(len(labels_non_onehot))])
    for i in range(len(labels)):
        labels[i] = np.eye(10)[labels_non_onehot[i]]

    test_labels = np.mat([[0 for i in range(10)] for k in range(len(test_labels_non_onehot))])
    for i in range(len(test_labels)):
        test_labels[i] = np.eye(10)[test_labels_non_onehot[i]]

    # test_labels = np.eye(10)[test_labels_non_onehot]

    nn = layer()
    batchsize = 100
    batch_xs = np.mat([[0.0 for n in range(3072)] for k in range(batchsize)])
    # batch_xs = np.mat([data[0] for k in range(batchsize)])
    print(labels[0])
    batch_ts = np.mat([[0 for n in range(10)] for k in range(batchsize)])
    print(test_data[0].shape)
    print(test_data.shape)

    i = 0
    for _ in range(20000):
        i += 1
        for n in range(batchsize):
            tmp = int(random.uniform(0, len(data)))
            batch_xs[n] = data[tmp].reshape(1, 3072)
            batch_xs[n] /= batch_xs[n].max()
            batch_ts[n] = labels[tmp].reshape(1, 10)
            # print(batch_xs[n])
            # print(batch_ts[n])
            # print(data[tmp].reshape(1, 3072))
            # plt.imshow(data[tmp].reshape(3, 32, 32).transpose(1, 2, 0))
            # plt.imshow(batch_xs[n].reshape(3, 32, 32).transpose(1, 2, 0))
            # plt.show()
        nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: test_data, nn.t: test_labels})
            print('Step: %d, Loss: %f, Accuracy: %f'
                  % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)

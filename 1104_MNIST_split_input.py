import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import load_cifer10
import random
from PIL import Image

# import load_notMNIST_1031

np.random.seed(20160612)
tf.set_random_seed(20160612)

image_size = 196


# ネットワーク構成
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('feed_forword'):
            with tf.name_scope('keep_prob'):
                keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('node_size'):
                with tf.name_scope('num_units3'):
                    num_units3 = 50
                with tf.name_scope('num_units2'):
                    num_units2 = 50
                with tf.name_scope('num_units1'):
                    num_units1 = 50

            with tf.name_scope('input_layer'):
                with tf.name_scope('input_size'):
                    input_size = image_size
                with tf.name_scope('x'):
                    x = tf.placeholder(tf.float32, [None, input_size])
                with tf.name_scope('x1'):
                    x1 = tf.placeholder(tf.float32, [None, input_size])
                with tf.name_scope('x2'):
                    x2 = tf.placeholder(tf.float32, [None, input_size])
                with tf.name_scope('x3'):
                    x3 = tf.placeholder(tf.float32, [None, input_size])

            with tf.name_scope('layer3'):
                with tf.name_scope('w3'):
                    w3 = tf.Variable(
                        tf.truncated_normal(shape=[input_size, num_units3], mean=0.0, stddev=1 / input_size))
                with tf.name_scope('w3_sub'):
                    w3_sub = tf.Variable(
                        tf.truncated_normal(shape=[input_size, num_units3], mean=0.0, stddev=1 / input_size))
                with tf.name_scope('b3'):
                    b3 = tf.Variable(tf.constant(0.1, shape=[num_units3]))
                with tf.name_scope('hidden3'):
                    hidden3 = tf.nn.relu(tf.matmul(x, w3) + b3 + tf.matmul(x1, w3_sub))
                with tf.name_scope('hidden8__drop'):
                    hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

            with tf.name_scope('layer2'):
                with tf.name_scope('w2'):
                    w2 = tf.Variable(
                        tf.truncated_normal(shape=[num_units3, num_units2], mean=0.0, stddev=1 / num_units3))
                with tf.name_scope('w2_sub'):
                    w2_sub = tf.Variable(
                        tf.truncated_normal(shape=[input_size, num_units2], mean=0.0, stddev=1 / num_units3))
                with tf.name_scope('b2'):
                    b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
                with tf.name_scope('hidden2'):
                    hidden2 = tf.nn.relu(tf.matmul(hidden3_drop, w2) + b2 + tf.matmul(x2, w2_sub))
                with tf.name_scope('hidden2_drop'):
                    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

            with tf.name_scope('layer1_fully_connected'):
                with tf.name_scope('w1'):
                    w1 = tf.Variable(
                        tf.truncated_normal(shape=[num_units2, num_units1], mean=0.0, stddev=1 / num_units2))
                with tf.name_scope('w1_sub'):
                    w1_sub = tf.Variable(
                        tf.truncated_normal(shape=[input_size, num_units1], mean=0.0, stddev=1 / num_units2))
                with tf.name_scope('b1'):
                    b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))
                with tf.name_scope('hidden1'):
                    hidden1 = tf.nn.relu(tf.matmul(hidden2_drop, w1) + b1 + tf.matmul(x3, w1_sub))
                with tf.name_scope('hidden1_drop'):
                    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

            with tf.name_scope('layer0_output_fully_connected'):
                with tf.name_scope('w0'):
                    w0 = tf.Variable(tf.truncated_normal(shape=[num_units1, 10], mean=0.0, stddev=1 / num_units1))
                with tf.name_scope('b0'):
                    b0 = tf.Variable(tf.zeros([10]))

                with tf.name_scope('p'):
                    p = tf.nn.softmax(b0 + tf.matmul(hidden1_drop, w0))

            t = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss'):
                loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
            # AdamOptimizer
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer(0.000001).minimize(loss)
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("accuracy", tf.reduce_sum(accuracy))
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w1", tf.reduce_sum(w1))
        tf.summary.scalar("w2", tf.reduce_sum(w2))
        tf.summary.scalar("w3", tf.reduce_sum(w3))

        self.x, self.t = x, t
        self.loss = loss
        self.accuracy = accuracy
        self.train_step = train_step
        self.keep_prob = keep_prob
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/1104_MNIST_split_input", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    nn = layer()
    batchsize = 1
    image_size = 196
    batch_xs_0 = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_xs_1 = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_xs_2 = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_xs_3 = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])

    test_xs_0 = np.mat([[0.0 for n in range(image_size)] for k in range(len(mnist.test.images))])
    test_xs_1 = np.mat([[0.0 for n in range(image_size)] for k in range(len(mnist.test.images))])
    test_xs_2 = np.mat([[0.0 for n in range(image_size)] for k in range(len(mnist.test.images))])
    test_xs_3 = np.mat([[0.0 for n in range(image_size)] for k in range(len(mnist.test.images))])

    for n in range(len(mnist.test.images)):
        test_xs_0[n] = mnist.train.images[n].reshape(28, 28)[0:14, 0:14].reshape(1, image_size)
        test_xs_1[n] = mnist.train.images[n].reshape(28, 28)[0:14, 14:28].reshape(1, image_size)
        test_xs_2[n] = mnist.train.images[n].reshape(28, 28)[14:28, 0:14].reshape(1, image_size)
        test_xs_3[n] = mnist.train.images[n].reshape(28, 28)[14:28, 14:28].reshape(1, image_size)
        # test_xs_0[n] /= 255  # test_xs_0[n].max()
        # test_xs_1[n] /= 255  # test_xs_1[n].max()
        # test_xs_2[n] /= 255  # test_xs_2[n].max()
        # test_xs_3[n] /= 255  # test_xs_3[n].max()
    print(test_xs_0[0])
    print(test_xs_1[0])
    print(test_xs_2[0])
    print(test_xs_3[0])
    loop_len = 100000
    print("step:{0} start".format(0))
    for i in range(loop_len):
        for n in range(batchsize):
            tmp = int(random.uniform(0, len(mnist.train.images)))
            batch_xs_0[n] = mnist.train.images[tmp].reshape(28, 28)[0:14, 0:14].reshape(1, image_size)
            batch_xs_1[n] = mnist.train.images[tmp].reshape(28, 28)[0:14, 14:28].reshape(1, image_size)
            batch_xs_2[n] = mnist.train.images[tmp].reshape(28, 28)[14:28, 0:14].reshape(1, image_size)
            batch_xs_3[n] = mnist.train.images[tmp].reshape(28, 28)[14:28, 14:28].reshape(1, image_size)

            # batch_xs_0[n] /= 255  # batch_xs_0[n].max()
            # batch_xs_1[n] /= 255  # batch_xs_1[n].max()
            # batch_xs_2[n] /= 255  # batch_xs_2[n].max()
            # batch_xs_3[n] /= 255  # batch_xs_3[n].max()

            batch_ts[n] = mnist.train.labels[tmp]  # .reshape(1, 10)
        nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs_1, nn.t: batch_ts, nn.keep_prob: 0.5,
                                              nn.x1: batch_xs_2, nn.x2: batch_xs_3, nn.x3: batch_xs_0})
        if i % 100 == 0:
            # train_data
            """
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.sess.run(nn.assign_op_train_total_accuracy,
                        feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
            """
            # test_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: test_xs_1, nn.t: mnist.test.labels, nn.keep_prob: 1.0,
                           nn.x1: test_xs_2, nn.x2: test_xs_3, nn.x3: test_xs_0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)
        i += 1

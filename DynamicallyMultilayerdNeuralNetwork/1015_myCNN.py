import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import load_cifer10
import random
from PIL import Image

np.random.seed(20160612)
tf.set_random_seed(20160612)

# ネットワーク構成
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input_layer'):
            with tf.name_scope('input_size'):
                input_size = 3072
            with tf.name_scope('num_filter1'):
                num_filters1 = 32
            with tf.name_scope('x'):
                x = tf.placeholder(tf.float32, [None, input_size])
            with tf.name_scope('x_image'):
                x_image = tf.reshape(x, [-1, 32, 32, 3])
        with tf.name_scope('conv_layer1'):
            with tf.name_scope('W_conv1'):
                W_conv1 = tf.Variable(tf.truncated_normal([4, 4, 3, num_filters1],
                                                          stddev=0.1))
            with tf.name_scope('h_conv1'):
                h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                                       strides=[1, 1, 1, 1], padding='SAME')

            with tf.name_scope('b_conv1'):
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
            with tf.name_scope('h_conv_cutoff'):
                h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

            with tf.name_scope('h_pool1'):
                h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('conv_layer2'):
            with tf.name_scope('num_filter2'):
                num_filters2 = 64
            with tf.name_scope('W_conv2'):
                W_conv2 = tf.Variable(
                    tf.truncated_normal([5, 5, num_filters1, num_filters2],
                                        stddev=0.1))
            with tf.name_scope('h_conv2'):
                h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                                       strides=[1, 1, 1, 1], padding='SAME')

            with tf.name_scope('b_conv2'):
                b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
            with tf.name_scope('h_conv2'):
                h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

            with tf.name_scope('h_pool2'):
                h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('h_pool2_flat'):
                h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * num_filters2])

        with tf.name_scope('layer3_fully_connected'):
            with tf.name_scope('num_units1'):
                num_units1 = 8 * 8 * num_filters2
            with tf.name_scope('num_units2'):
                num_units2 = 1024

            with tf.name_scope('w2'):
                w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
            with tf.name_scope('hidden2'):
                hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

            with tf.name_scope('keep_prob'):
                keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('hidden2_drop'):
                hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        with tf.name_scope('layer4_output_fully_connected'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(tf.zeros([num_units2, 10]))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([10]))
            with tf.name_scope('p'):
                p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

            t = tf.placeholder(tf.float32, [None, 10])
        with tf.name_scope('optimizer'):
            with tf.name_scope('loss'):
                loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w2", tf.reduce_sum(w2))

        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        self.keep_prob = keep_prob
        # self.w2 = w2
        # self.w1 = w1

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/cifar10_logs", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    data, labels_non_onehot, test_data, test_labels_non_onehot, label_names = load_cifer10.load_dataset()
    print(label_names)
    # labels = np.eye(10)[labels_non_onehot]
    # test_labels = np.eye(10)[test_labels_non_onehot]

    labels = np.mat([[0 for i in range(10)] for k in range(len(labels_non_onehot))])
    for i in range(len(labels)):
        labels[i] = np.eye(10)[labels_non_onehot[i]]
    test_labels = np.mat([[0 for i in range(10)] for k in range(len(test_labels_non_onehot))])
    for i in range(len(test_labels)):
        test_labels[i] = np.eye(10)[test_labels_non_onehot[i]]

    nn = layer()
    batchsize = 100
    # batch_xs = np.mat([data[0] for k in range(batchsize)])
    # batch_ts = np.mat([labels[0] for k in range(batchsize)])
    batch_xs = np.mat([[0.0 for n in range(3072)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])
    print(test_data[0].shape)
    print(test_data.shape)

    i = 0
    for _ in range(1000):
        i += 1
        for n in range(batchsize):
            tmp = int(random.uniform(0, len(data)))
            batch_xs[n] = data[tmp].reshape(1, 3072)
            batch_xs[n] /= batch_xs[n].max()
            batch_ts[n] = labels[tmp].reshape(1, 10)
            # print(batch_xs[n])
            # print(batch_ts[n])
            # plt.imshow(data[tmp].reshape(3, 32, 32).transpose(1, 2, 0))
            # plt.imshow(batch_xs[n].reshape(3, 32, 32).transpose(1, 2, 0))
            # plt.show()
        nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: 0.5})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f'
                  % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)

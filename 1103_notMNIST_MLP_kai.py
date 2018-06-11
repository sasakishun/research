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

image_size = 784


# ネットワーク構成
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input_layer'):
            with tf.name_scope('input_size'):
                input_size = image_size
            with tf.name_scope('x'):
                x = tf.placeholder(tf.float32, [None, input_size])

        with tf.name_scope('keep_prob'):
            keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('node_size'):
            with tf.name_scope('num_units7'):
                num_units7 = 600
            with tf.name_scope('num_units6'):
                num_units6 = 600
            with tf.name_scope('num_units5'):
                num_units5 = 600
            with tf.name_scope('num_units4'):
                num_units4 = 600
            with tf.name_scope('num_units3'):
                num_units3 = 600
            with tf.name_scope('num_units2'):
                num_units2 = 600
            with tf.name_scope('num_units1'):
                num_units1 = 600

        with tf.name_scope('layer7'):
            with tf.name_scope('w7'):
                w7 = tf.Variable(tf.truncated_normal(shape=[input_size, num_units7], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('b7'):
                b7 = tf.Variable(tf.constant(0.1, shape=[num_units7]))

        with tf.name_scope('layer6'):
            with tf.name_scope('w6'):
                w6 = tf.Variable(tf.truncated_normal(shape=[num_units7, num_units6], mean=0.0, stddev=1 / num_units7))
            with tf.name_scope('b6'):
                b6 = tf.Variable(tf.constant(0.1, shape=[num_units6]))

        with tf.name_scope('layer5'):
            with tf.name_scope('w5'):
                w5 = tf.Variable(tf.truncated_normal(shape=[num_units6, num_units5], mean=0.0, stddev=1 / num_units6))
            with tf.name_scope('b5'):
                b5 = tf.Variable(tf.constant(0.1, shape=[num_units5]))

        with tf.name_scope('layer4'):
            with tf.name_scope('w4'):
                w4 = tf.Variable(tf.truncated_normal(shape=[num_units5, num_units4], mean=0.0, stddev=1 / num_units5))
            with tf.name_scope('b4'):
                b4 = tf.Variable(tf.constant(0.1, shape=[num_units4]))

        with tf.name_scope('layer3'):
            with tf.name_scope('w2'):
                w3 = tf.Variable(tf.truncated_normal(shape=[num_units4, num_units3], mean=0.0, stddev=1 / num_units4))
            with tf.name_scope('b2'):
                b3 = tf.Variable(tf.constant(0.1, shape=[num_units3]))

        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w2 = tf.Variable(tf.truncated_normal(shape=[num_units3, num_units2], mean=0.0, stddev=1 / num_units3))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))

        with tf.name_scope('layer1_fully_connected'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(tf.truncated_normal(shape=[num_units2, num_units1], mean=0.0, stddev=1 / num_units2))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))

        with tf.name_scope('layer0_output_fully_connected'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(tf.truncated_normal(shape=[num_units1, 10], mean=0.0, stddev=1 / num_units1))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('feed_forword'):
            with tf.name_scope('hidden8_7'):
                hidden8_7 = tf.nn.relu(tf.matmul(x, w7) + b7)
            with tf.name_scope('hidden8__drop'):
                hidden8_7_drop = tf.nn.dropout(hidden8_7, keep_prob)

            with tf.name_scope('hidden8_6'):
                hidden8_6 = tf.nn.relu(tf.matmul(hidden8_7_drop, w6) + b6)
            with tf.name_scope('hidden8_6_drop'):
                hidden8_6_drop = tf.nn.dropout(hidden8_6, keep_prob)

            with tf.name_scope('hidden8_5'):
                hidden8_5 = tf.nn.relu(tf.matmul(hidden8_6_drop, w5) + b5)
            with tf.name_scope('hidden8__drop'):
                hidden8_5_drop = tf.nn.dropout(hidden8_5, keep_prob)

            with tf.name_scope('hidden8_4'):
                hidden8_4 = tf.nn.relu(tf.matmul(hidden8_5_drop, w4) + b4)
            with tf.name_scope('hidden8__drop'):
                hidden8_4_drop = tf.nn.dropout(hidden8_4, keep_prob)

            with tf.name_scope('hidden8_3'):
                hidden8_3 = tf.nn.relu(tf.matmul(hidden8_4_drop, w3) + b3)
            with tf.name_scope('hidden8__drop'):
                hidden8_3_drop = tf.nn.dropout(hidden8_3, keep_prob)

            with tf.name_scope('hidden8_2'):
                hidden8_2 = tf.nn.relu(tf.matmul(hidden8_3_drop, w2) + b2)
            with tf.name_scope('hidden8__drop'):
                hidden8_2_drop = tf.nn.dropout(hidden8_2, keep_prob)

            with tf.name_scope('hidden8_1'):
                hidden8_1 = tf.nn.relu(tf.matmul(hidden8_2_drop, w1) + b1)
            with tf.name_scope('hidden8_1_drop'):
                hidden8_1_drop = tf.nn.dropout(hidden8_1, keep_prob)
            with tf.name_scope('p8'):
                p8 = tf.nn.softmax(b0 + tf.matmul(hidden8_1_drop, w0))

            t = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss'):
                loss = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p8, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_3, 1e-10, 1.0)))

            """
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer(0.0001).minimize(loss0, var_list=[w0_9, b0])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1, var_list=[w8, b8, w0_8, b0])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2, var_list=[w8, b8, w7, b7, w0_7, b0])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer(0.0001).minimize(loss3,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w0_6, b0])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer(0.0001).minimize(loss4,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w0_5,
                                                                                b0])
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdamOptimizer(0.0001).minimize(loss5,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w0_4, b0])
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.AdamOptimizer(0.0001).minimize(loss6,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, w0_3, b0])
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.AdamOptimizer(0.0001).minimize(loss7,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, w2, b2, w0_2, b0])
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.AdamOptimizer(0.0001).minimize(loss8,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, w2, b2, w1, b1, w0, b0])
            """
            """
            with tf.name_scope('train_step0'):
                eta = 0.001
                train_step0 = tf.train.GradientDescentOptimizer(eta).minimize(loss0, var_list=[w0_9, b0])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.GradientDescentOptimizer(eta).minimize(loss1, var_list=[w8, b8, b0])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.GradientDescentOptimizer(eta).minimize(loss2, var_list=[w8, b8, w7, b7, b0])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.GradientDescentOptimizer(eta).minimize(loss3,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, b0])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.GradientDescentOptimizer(eta).minimize(loss4,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5,
                                                                                b0])
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.GradientDescentOptimizer(eta).minimize(loss5,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                b0])
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.GradientDescentOptimizer(eta).minimize(loss6,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, b0])
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.GradientDescentOptimizer(eta).minimize(loss7,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, w2, b2, b0])
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.GradientDescentOptimizer(eta).minimize(loss8,
                                                                      var_list=[w8, b8, w7, b7, w6, b6, w5, b5, w4, b4,
                                                                                w3, b3, w2, b2, w1, b1, w0, b0])

            """
            # AdamOptimizer
            with tf.name_scope('train_step8'):
                train_step = tf.train.AdamOptimizer(0.0001).minimize(loss,
                                                                     var_list=[w7, b7, w6, b6, w5, b5, w4, b4,
                                                                               w3, b3, w2, b2, w1, b1, w0, b0])
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p8, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('assign_input'):
            """
            input_placeholder0_9 = tf.placeholder(tf.float32, shape=[input_size, 10])
            input_placeholder0_8 = tf.placeholder(tf.float32, shape=[num_units8, 10])
            input_placeholder0_7 = tf.placeholder(tf.float32, shape=[num_units7, 10])
            input_placeholder0_6 = tf.placeholder(tf.float32, shape=[num_units6, 10])
            input_placeholder0_5 = tf.placeholder(tf.float32, shape=[num_units5, 10])
            input_placeholder0_4 = tf.placeholder(tf.float32, shape=[num_units4, 10])
            input_placeholder0_3 = tf.placeholder(tf.float32, shape=[num_units3, 10])
            input_placeholder0_2 = tf.placeholder(tf.float32, shape=[num_units2, 10])

            assign_op0_9 = w0_9.assign(input_placeholder0_9)
            assign_op0_8 = w0_8.assign(input_placeholder0_8)
            assign_op0_7 = w0_7.assign(input_placeholder0_7)
            assign_op0_6 = w0_6.assign(input_placeholder0_6)
            assign_op0_5 = w0_5.assign(input_placeholder0_5)
            assign_op0_4 = w0_4.assign(input_placeholder0_4)
            assign_op0_3 = w0_3.assign(input_placeholder0_3)
            assign_op0_2 = w0_2.assign(input_placeholder0_2)
            """
            total_accuracy = tf.Variable(0.)
            input_placeholder_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_total_accuracy = total_accuracy.assign(input_placeholder_total_accuracy)

            train_total_accuracy = tf.Variable(0.)
            input_placeholder_train_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_train_total_accuracy = train_total_accuracy.assign(input_placeholder_train_total_accuracy)

        tf.summary.scalar("accuracy", tf.reduce_sum(accuracy))
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w1", tf.reduce_sum(w1))
        tf.summary.scalar("w2", tf.reduce_sum(w2))
        tf.summary.scalar("w3", tf.reduce_sum(w3))
        tf.summary.scalar("w4", tf.reduce_sum(w4))
        tf.summary.scalar("w5", tf.reduce_sum(w5))
        tf.summary.scalar("w6", tf.reduce_sum(w6))
        tf.summary.scalar("w7", tf.reduce_sum(w7))
        # tf.summary.scalar("w8", tf.reduce_sum(w8))

        tf.summary.scalar("total_accuracy", tf.reduce_sum(total_accuracy))
        tf.summary.scalar("train_total_accuracy", tf.reduce_sum(train_total_accuracy))

        self.x, self.t = x, t
        self.loss = loss
        self.accuracy = accuracy
        self.total_accuracy = total_accuracy
        self.train_step = train_step
        self.keep_prob = keep_prob
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        # self.w8 = w8

        self.assign_op_total_accuracy = assign_op_total_accuracy
        self.input_placeholder_total_accuracy = input_placeholder_total_accuracy

        self.assign_op_train_total_accuracy = assign_op_train_total_accuracy
        self.input_placeholder_train_total_accuracy = input_placeholder_train_total_accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/1103_notMNIST_MLP_kai", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    # data, test_data, labels_non_onehot, test_labels_non_onehot = load_notMNIST_1031.get_data()
    # data, labels_non_onehot, test_data, test_labels_non_onehot, label_names = load_cifer10.load_dataset()
    data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    labels = np.load("train_label.npy")
    test_labels = np.load("test_label.npy")

    """
    labels = np.mat([[0 for i in range(10)] for k in range(len(labels_non_onehot))])
    for i in range(len(labels)):
        labels[i] = np.eye(10)[int(labels_non_onehot[i])]
    test_labels = np.mat([[0 for i in range(10)] for k in range(len(test_labels_non_onehot))])
    for i in range(len(test_labels)):
        test_labels[i] = np.eye(10)[test_labels_non_onehot[i]]
    """
    # np.save("train_data", data)
    # np.save("train_label", labels)
    # np.save("test_data", test_data)
    # np.save("test_label", test_labels)

    nn = layer()
    batchsize = 10
    batch_xs = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])
    print(test_data[0].shape)
    print(test_data.shape)

    train_list_num = list(range(len(data)))
    train_list_num = random.sample(train_list_num, 10000)
    train_test_data = np.mat([[0.0 for n in range(image_size)] for k in range(len(train_list_num))])
    train_test_labels = np.mat([[0.0 for n in range(10)] for k in range(len(train_list_num))])

    for i in range(len(train_test_data)):
        tmp = train_list_num[i]
        train_test_data[i] = data[tmp].reshape(1, image_size)
        # train_test_data[i] /= train_test_data[i].max()
        train_test_labels[i] = labels[tmp].reshape(1, 10)

    loop_len = 3200000
    # decay_rate = 0.99999
    # print("decay_rate:{0}".format(decay_rate))
    print("step:{0} start".format(0))
    for i in range(loop_len):
        for n in range(batchsize):
            tmp = int(random.uniform(0, len(data)))
            batch_xs[n] = data[tmp].reshape(1, image_size)
            # batch_xs[n] /= batch_xs[n].max()
            batch_ts[n] = labels[tmp].reshape(1, 10)
        nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: 0.5})
        if i % 100 == 0:
            # train_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.sess.run(nn.assign_op_train_total_accuracy,
                        feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
            # test_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.sess.run(nn.assign_op_total_accuracy, feed_dict={nn.input_placeholder_total_accuracy: acc_val})
        nn.writer.add_summary(summary, i)
        i += 1

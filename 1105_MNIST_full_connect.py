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
        with tf.name_scope('node_size_200'):
            with tf.name_scope('num_units5'):
                num_units5 = 200
            with tf.name_scope('num_units4'):
                num_units4 = 200
            with tf.name_scope('num_units3'):
                num_units3 = 200
            with tf.name_scope('num_units2'):
                num_units2 = 200
            with tf.name_scope('num_units1'):
                num_units1 = 200

        with tf.name_scope('layer5'):
            with tf.name_scope('w5'):
                w5 = tf.Variable(tf.truncated_normal(shape=[input_size, num_units5], mean=0.0, stddev=1 / input_size))
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

        with tf.name_scope('layer1'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(tf.truncated_normal(shape=[num_units2, num_units1], mean=0.0, stddev=1 / num_units2))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))

        with tf.name_scope('layer0'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(tf.truncated_normal(shape=[num_units1, 10], mean=0.0, stddev=1 / num_units1))
            with tf.name_scope('w5_9'):
                w0_5 = tf.Variable(tf.truncated_normal(shape=[num_units5, 10], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('w4_9'):
                w0_4 = tf.Variable(tf.truncated_normal(shape=[num_units4, 10], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('w3_9'):
                w0_3 = tf.Variable(tf.truncated_normal(shape=[num_units3, 10], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('w2_9'):
                w0_2 = tf.Variable(tf.truncated_normal(shape=[num_units2, 10], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('feed_forword'):
            with tf.name_scope('state0'):
                with tf.name_scope('p0'):
                    p0 = tf.nn.softmax(tf.matmul(x, w0_9) + b0)

            with tf.name_scope('state1'):
                with tf.name_scope('hidden1_1'):
                    hidden1_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden1_1_drop'):
                    hidden1_8_drop = tf.nn.dropout(hidden1_8, keep_prob)
                with tf.name_scope('p1'):
                    p1 = tf.nn.softmax(tf.matmul(hidden1_8_drop, w0_8) + b0 + tf.matmul(x, w0_9))

            with tf.name_scope('state2'):
                with tf.name_scope('hidden2_2'):
                    hidden2_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden2_2_drop'):
                    hidden2_8_drop = tf.nn.dropout(hidden2_8, keep_prob)
                with tf.name_scope('hidden2_1'):
                    hidden2_7 = tf.nn.relu(tf.matmul(hidden2_8_drop, w7) + b7)
                with tf.name_scope('hidden2_1_drop'):
                    hidden2_7_drop = tf.nn.dropout(hidden2_7, keep_prob)
                with tf.name_scope('p2'):
                    p2 = tf.nn.softmax(tf.matmul(hidden2_7_drop, w0_7) + b0
                                       + tf.matmul(x, w0_9)
                                       + tf.matmul(hidden2_7, w0_7))

            with tf.name_scope('state3'):
                with tf.name_scope('hidden3_3'):
                    hidden3_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden3_3_drop'):
                    hidden3_8_drop = tf.nn.dropout(hidden3_8, keep_prob)
                with tf.name_scope('hidden3_2'):
                    hidden3_7 = tf.nn.relu(tf.matmul(hidden3_8_drop, w7) + b7)
                with tf.name_scope('hidden3_2_drop'):
                    hidden3_7_drop = tf.nn.dropout(hidden3_7, keep_prob)
                with tf.name_scope('hidden3_1'):
                    hidden3_6 = tf.nn.relu(tf.matmul(hidden3_7_drop, w6) + b6)
                with tf.name_scope('hidden3_1_drop'):
                    hidden3_6_drop = tf.nn.dropout(hidden3_6, keep_prob)
                with tf.name_scope('p3'):
                    p3 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden3_6_drop, w0_6)
                                       + tf.matmul(hidden3_7_drop, w0_7)
                                       + tf.matmul(hidden3_8_drop, w0_8))

            with tf.name_scope('state4'):
                with tf.name_scope('hidden4_4'):
                    hidden4_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden4_4_drop'):
                    hidden4_8_drop = tf.nn.dropout(hidden4_8, keep_prob)

                with tf.name_scope('hidden4_3'):
                    hidden4_7 = tf.nn.relu(tf.matmul(hidden4_8_drop, w7) + b7)
                with tf.name_scope('hidden4_3_drop'):
                    hidden4_7_drop = tf.nn.dropout(hidden4_7, keep_prob)

                with tf.name_scope('hidden4_2'):
                    hidden4_6 = tf.nn.relu(tf.matmul(hidden4_7_drop, w6) + b6)
                with tf.name_scope('hidden4_2_drop'):
                    hidden4_6_drop = tf.nn.dropout(hidden4_6, keep_prob)

                with tf.name_scope('hidden4_1'):
                    hidden4_5 = tf.nn.relu(tf.matmul(hidden4_6_drop, w5) + b5)
                with tf.name_scope('hidden4_1_drop'):
                    hidden4_5_drop = tf.nn.dropout(hidden4_5, keep_prob)
                with tf.name_scope('p4'):
                    p4 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden4_5_drop, w0_5)
                                       + tf.matmul(hidden4_6_drop, w0_6)
                                       + tf.matmul(hidden4_7_drop, w0_7)
                                       + tf.matmul(hidden4_8_drop, w0_8))

            with tf.name_scope('state5'):
                with tf.name_scope('hidden5_5'):
                    hidden5_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden5_5_drop'):
                    hidden5_8_drop = tf.nn.dropout(hidden5_8, keep_prob)

                with tf.name_scope('hidden5_4'):
                    hidden5_7 = tf.nn.relu(tf.matmul(hidden5_8_drop, w7) + b7)
                with tf.name_scope('hidden5_4_drop'):
                    hidden5_7_drop = tf.nn.dropout(hidden5_7, keep_prob)

                with tf.name_scope('hidden5_3'):
                    hidden5_6 = tf.nn.relu(tf.matmul(hidden5_7_drop, w6) + b6)
                with tf.name_scope('hidden5_3_drop'):
                    hidden5_6_drop = tf.nn.dropout(hidden5_6, keep_prob)

                with tf.name_scope('hidden5_2'):
                    hidden5_5 = tf.nn.relu(tf.matmul(hidden5_6_drop, w5) + b5)
                with tf.name_scope('hidden5_2_drop'):
                    hidden5_5_drop = tf.nn.dropout(hidden5_5, keep_prob)

                with tf.name_scope('hidden5_1'):
                    hidden5_4 = tf.nn.relu(tf.matmul(hidden5_5_drop, w4) + b4)
                with tf.name_scope('hidden5_1_drop'):
                    hidden5_4_drop = tf.nn.dropout(hidden5_4, keep_prob)
                with tf.name_scope('p5'):
                    p5 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden5_4_drop, w0_4)
                                       + tf.matmul(hidden5_5_drop, w0_5)
                                       + tf.matmul(hidden5_6_drop, w0_6)
                                       + tf.matmul(hidden5_7_drop, w0_7)
                                       + tf.matmul(hidden5_8_drop, w0_8))

            t = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss0'):
                loss0 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0)))
            with tf.name_scope('loss1'):
                loss1 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0)))
            with tf.name_scope('loss2'):
                loss2 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p2, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_9, 1e-10, 1.0)))
            with tf.name_scope('loss3'):
                loss3 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p3, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_8, 1e-10, 1.0)))
            with tf.name_scope('loss4'):
                loss4 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p4, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_7, 1e-10, 1.0)))
            with tf.name_scope('loss5'):
                loss5 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p5, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_6, 1e-10, 1.0)))
            loss = [loss0, loss1, loss2, loss3, loss4, loss5]

            # AdamOptimizer
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer(0.0001).minimize(loss0)
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer(0.0001).minimize(loss3)
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer(0.0001).minimize(loss4)
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdamOptimizer(0.0001).minimize(loss5)
            train_step = [train_step0, train_step1, train_step2, train_step3, train_step4, train_step5]
            with tf.name_scope('correct_prediction'):
                correct_prediction0 = tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1))
                correct_prediction1 = tf.equal(tf.argmax(p1, 1), tf.argmax(t, 1))
                correct_prediction2 = tf.equal(tf.argmax(p2, 1), tf.argmax(t, 1))
                correct_prediction3 = tf.equal(tf.argmax(p3, 1), tf.argmax(t, 1))
                correct_prediction4 = tf.equal(tf.argmax(p4, 1), tf.argmax(t, 1))
                correct_prediction5 = tf.equal(tf.argmax(p5, 1), tf.argmax(t, 1))
                correct_prediction6 = tf.equal(tf.argmax(p6, 1), tf.argmax(t, 1))
                correct_prediction7 = tf.equal(tf.argmax(p7, 1), tf.argmax(t, 1))
                correct_prediction8 = tf.equal(tf.argmax(p8, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
                accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
                accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
                accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
                accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
                accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
                accuracy = [accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]

        with tf.name_scope('assign_input'):
            input_placeholder0_5 = tf.placeholder(tf.float32, shape=[num_units5, 10])
            input_placeholder0_4 = tf.placeholder(tf.float32, shape=[num_units4, 10])
            input_placeholder0_3 = tf.placeholder(tf.float32, shape=[num_units3, 10])
            input_placeholder0_2 = tf.placeholder(tf.float32, shape=[num_units2, 10])

            assign_op0_5 = w0_5.assign(input_placeholder0_5)
            assign_op0_4 = w0_4.assign(input_placeholder0_4)
            assign_op0_3 = w0_3.assign(input_placeholder0_3)
            assign_op0_2 = w0_2.assign(input_placeholder0_2)

            total_accuracy = tf.Variable(0.)
            input_placeholder_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_total_accuracy = total_accuracy.assign(input_placeholder_total_accuracy)

            train_total_accuracy = tf.Variable(0.)
            input_placeholder_train_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_train_total_accuracy = train_total_accuracy.assign(input_placeholder_train_total_accuracy)

        tf.summary.scalar("accuracy0", tf.reduce_sum(accuracy[0]))
        tf.summary.scalar("accuracy1", tf.reduce_sum(accuracy[1]))
        tf.summary.scalar("accuracy2", tf.reduce_sum(accuracy[2]))
        tf.summary.scalar("accuracy3", tf.reduce_sum(accuracy[3]))
        tf.summary.scalar("accuracy4", tf.reduce_sum(accuracy[4]))
        tf.summary.scalar("accuracy5", tf.reduce_sum(accuracy[5]))
        tf.summary.scalar("w0", tf.reduce_sum(w0))
        tf.summary.scalar("w1", tf.reduce_sum(w1))
        tf.summary.scalar("w2", tf.reduce_sum(w2))
        tf.summary.scalar("w3", tf.reduce_sum(w3))
        tf.summary.scalar("w4", tf.reduce_sum(w4))
        tf.summary.scalar("w5", tf.reduce_sum(w5))

        tf.summary.scalar("w0_5", tf.reduce_sum(w0_5))
        tf.summary.scalar("w0_4", tf.reduce_sum(w0_4))
        tf.summary.scalar("w0_3", tf.reduce_sum(w0_3))
        tf.summary.scalar("w0_2", tf.reduce_sum(w0_2))
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
        self.w0_5 = w0_5
        self.w0_4 = w0_4
        self.w0_3 = w0_3
        self.w0_2 = w0_2
        self.assign_op0_5 = assign_op0_5
        self.assign_op0_4 = assign_op0_4
        self.assign_op0_3 = assign_op0_3
        self.assign_op0_2 = assign_op0_2
        self.input_placeholder0_5 = input_placeholder0_5
        self.input_placeholder0_4 = input_placeholder0_4
        self.input_placeholder0_3 = input_placeholder0_3
        self.input_placeholder0_2 = input_placeholder0_2

        self.assign_op_total_accuracy = assign_op_total_accuracy
        self.input_placeholder_total_accuracy = input_placeholder_total_accuracy

        self.assign_op_train_total_accuracy = assign_op_train_total_accuracy
        self.input_placeholder_train_total_accuracy = input_placeholder_train_total_accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/1105_MNIST_nonliner", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    data = np.load("MNIST_train_data.npy")
    labels = np.load("MNIST_train_labels.npy")
    test_data = np.load("MNIST_test_data.npy")
    test_labels = np.load("MNIST_test_labels.npy")

    """
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    data = mnist.train.images
    labels = mnist.train.labels
    test_data = mnist.test.images
    test_labels = mnist.test.labels
    np.save("MNIST/MNIST_train_data", data)
    np.save("MNIST/MNIST_train_labels", labels)
    np.save("MNIST/MNIST_test_data", test_data)
    np.save("MNIST/MNIST_test_labels", test_labels)
    """
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

    loop_len = 10000  # 400000
    decay_rate = 0.999999
    print("decay_rate:{0}".format(decay_rate))
    for j in range(len(nn.train_step)):
        print("step:{0} start".format(j))
        for i in range(loop_len):
            for n in range(batchsize):
                tmp = int(random.uniform(0, len(data)))
                batch_xs[n] = data[tmp].reshape(1, image_size)
                # batch_xs[n] /= batch_xs[n].max()
                batch_ts[n] = labels[tmp].reshape(1, 10)
            nn.sess.run(nn.train_step[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: 0.5})
            if i % 100 == 0:
                # train_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
                print('Step: %d, Loss(tr): %f, Accuracy: %f' % (i + j * loop_len, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
                print('Step: %d, Loss(te): %f, Accuracy: %f' % (i + j * loop_len, loss_val, acc_val))
                nn.sess.run(nn.assign_op_total_accuracy, feed_dict={nn.input_placeholder_total_accuracy: acc_val})
            nn.writer.add_summary(summary, i + j * loop_len)
            i += 1
        j += 1

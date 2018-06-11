import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import load_cifer10
import random
from PIL import Image

# import load_notMNIST_1031

# np.random.seed(20160612)
# tf.set_random_seed(20160612)
np.random.seed(20171109)
tf.set_random_seed(20171109)

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
            with tf.name_scope('num_units8'):
                hidden_unit = 2000
            with tf.name_scope('num_units8'):
                num_units8 = 2500  # hidden_unit
            with tf.name_scope('num_units7'):
                num_units7 = hidden_unit
            with tf.name_scope('num_units6'):
                num_units6 = hidden_unit
            with tf.name_scope('num_units5'):
                num_units5 = hidden_unit
            with tf.name_scope('num_units4'):
                num_units4 = hidden_unit
            with tf.name_scope('num_units3'):
                num_units3 = hidden_unit
            with tf.name_scope('num_units2'):
                num_units2 = hidden_unit
            with tf.name_scope('num_units1'):
                num_units1 = hidden_unit

        with tf.name_scope('layer8'):
            with tf.name_scope('w8'):
                w8 = tf.Variable(tf.truncated_normal(shape=[input_size, num_units8], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('b8'):
                b8 = tf.Variable(tf.constant(0.1, shape=[num_units8]))

        with tf.name_scope('layer7'):
            with tf.name_scope('w7'):
                w7 = tf.Variable(tf.truncated_normal(shape=[num_units8, num_units7], mean=0.0, stddev=1 / num_units8))
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

        with tf.name_scope('layer1'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(tf.truncated_normal(shape=[num_units2, num_units1], mean=0.0, stddev=1 / num_units2))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))

        with tf.name_scope('layer0'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(tf.truncated_normal(shape=[num_units1, 10], mean=0.0, stddev=1 / num_units1))
            with tf.name_scope('w0_9'):
                w0_9 = tf.Variable(tf.truncated_normal(shape=[input_size, 10], mean=0.0, stddev=1 / input_size))
            with tf.name_scope('w7_9'):
                w0_8 = tf.Variable(tf.truncated_normal(shape=[num_units8, 10], mean=0.0, stddev=1 / num_units8))
            with tf.name_scope('w7_9'):
                w0_7 = tf.Variable(tf.truncated_normal(shape=[num_units7, 10], mean=0.0, stddev=1 / num_units7))
            with tf.name_scope('w6_9'):
                w0_6 = tf.Variable(tf.truncated_normal(shape=[num_units6, 10], mean=0.0, stddev=1 / num_units6))
            with tf.name_scope('w5_9'):
                w0_5 = tf.Variable(tf.truncated_normal(shape=[num_units5, 10], mean=0.0, stddev=1 / num_units5))
            with tf.name_scope('w4_9'):
                w0_4 = tf.Variable(tf.truncated_normal(shape=[num_units4, 10], mean=0.0, stddev=1 / num_units4))
            with tf.name_scope('w3_9'):
                w0_3 = tf.Variable(tf.truncated_normal(shape=[num_units3, 10], mean=0.0, stddev=1 / num_units3))
            with tf.name_scope('w2_9'):
                w0_2 = tf.Variable(tf.truncated_normal(shape=[num_units2, 10], mean=0.0, stddev=1 / num_units2))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('feed_forword'):
            with tf.name_scope('state8'):
                with tf.name_scope('hidden8_8'):
                    hidden8_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden8_8_drop'):
                    hidden8_8_drop = tf.nn.dropout(hidden8_8, keep_prob)

                with tf.name_scope('hidden8_7'):
                    hidden8_7 = tf.nn.relu(tf.matmul(hidden8_8_drop, w7) + b7)
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
                    p8 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden8_1_drop, w0)
                                       + tf.matmul(hidden8_2_drop, w0_2)
                                       + tf.matmul(hidden8_3_drop, w0_3)
                                       + tf.matmul(hidden8_4_drop, w0_4)
                                       + tf.matmul(hidden8_5_drop, w0_5)
                                       + tf.matmul(hidden8_6_drop, w0_6)
                                       + tf.matmul(hidden8_7_drop, w0_7)
                                       + tf.matmul(hidden8_8_drop, w0_8))

            t = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss8'):
                loss = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p8, 1e-10, 1.0)))
            learning_rate = 0.001  # 0.0001
            # AdamOptimizer
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # Adadelta
            """
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss0, var_list=[w0_9])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss1, var_list=[w0_8, w8, b8])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss2, var_list=[w0_7, w7, b7])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss3, var_list=[w0_6, w6, b6])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss4, var_list=[w0_5, w5, b5])
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss5, var_list=[w0_4, w4, b4])
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss6, var_list=[w0_3, w3, b3])
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss7, var_list=[w0_2, w2, b2])
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.AdadeltaOptimizer(0.0001).minimize(loss8, var_list=[w0, b0, w1, b1])
            """
            # Adam
            """
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer(0.0001).minimize(loss0, var_list=[w0_9])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1, var_list=[w0_8, w8, b8])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2, var_list=[w0_7, w7, b7])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer(0.0001).minimize(loss3, var_list=[w0_6, w6, b6])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer(0.0001).minimize(loss4, var_list=[w0_5, w5, b5])
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdamOptimizer(0.0001).minimize(loss5, var_list=[w0_4, w4, b4])
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.AdamOptimizer(0.0001).minimize(loss6, var_list=[w0_3, w3, b3])
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.AdamOptimizer(0.0001).minimize(loss7, var_list=[w0_2, w2, b2])
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.AdamOptimizer(0.0001).minimize(loss8, var_list=[w0, b0, w1, b1])
            """
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(p8, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('assign_input'):
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
        tf.summary.scalar("w8", tf.reduce_sum(w8))

        tf.summary.scalar("w0_9", tf.reduce_sum(w0_9))
        tf.summary.scalar("w0_8", tf.reduce_sum(w0_8))
        tf.summary.scalar("w0_7", tf.reduce_sum(w0_7))
        tf.summary.scalar("w0_6", tf.reduce_sum(w0_6))
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
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.w0_9 = w0_9
        self.w0_8 = w0_8
        self.w0_7 = w0_7
        self.w0_6 = w0_6
        self.w0_5 = w0_5
        self.w0_4 = w0_4
        self.w0_3 = w0_3
        self.w0_2 = w0_2
        self.assign_op0_9 = assign_op0_9
        self.assign_op0_8 = assign_op0_8
        self.assign_op0_7 = assign_op0_7
        self.assign_op0_6 = assign_op0_6
        self.assign_op0_5 = assign_op0_5
        self.assign_op0_4 = assign_op0_4
        self.assign_op0_3 = assign_op0_3
        self.assign_op0_2 = assign_op0_2
        self.input_placeholder0_9 = input_placeholder0_9
        self.input_placeholder0_8 = input_placeholder0_8
        self.input_placeholder0_7 = input_placeholder0_7
        self.input_placeholder0_6 = input_placeholder0_6
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
        # writer = tf.summary.FileWriter("./log/1105_MNIST_shortcutMLP", sess.graph)
        writer = tf.summary.FileWriter("./log/1109_compare/shortcut_MLP", sess.graph)
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
    MLP = __import__("1106_MNIST_MLP")
    mlp = MLP.layer()
    batchsize = 50
    batch_xs = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])
    print(test_data[0].shape)
    print(test_data.shape)
    train_size = 10000  # 10000 is good
    # train_list_num = list(range(len(data)))
    train_list_num = random.sample(list(range(len(data))), train_size)
    train_test_data = np.mat([[0.0 for n in range(image_size)] for k in range(len(train_list_num))])
    train_test_labels = np.mat([[0.0 for n in range(10)] for k in range(len(train_list_num))])

    for i in range(len(train_test_data)):
        tmp = train_list_num[i]
        train_test_data[i] = data[tmp].reshape(1, image_size)
        # train_test_data[i] /= train_test_data[i].max()
        train_test_labels[i] = labels[tmp].reshape(1, 10)

    loop_len = 200000  # 400000
    # decay_rate = 0.999999
    # print("decay_rate:{0}".format(decay_rate))
    for i in range(loop_len):
        for n in range(batchsize):
            # tmp = int(random.uniform(0, len(data)))
            tmp = int(random.uniform(0, train_size))
            batch_xs[n] = data[tmp].reshape(1, image_size)
            # batch_xs[n] /= batch_xs[n].max()
            batch_ts[n] = labels[tmp].reshape(1, 10)
        nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: 1.0})
        mlp.sess.run(mlp.train_step, feed_dict={mlp.x: batch_xs, mlp.t: batch_ts, mlp.keep_prob: 1.0})
        if i % 100 == 0:
            # train_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss[j], nn.accuracy[j]],
                feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss(tr): %f, Accuracy: %f' % (i + j * loop_len, loss_val, acc_val))
            nn.sess.run(nn.assign_op_train_total_accuracy,
                        feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})

            summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                [mlp.summary, mlp.loss, mlp.accuracy],
                feed_dict={mlp.x: train_test_data, mlp.t: train_test_labels, mlp.keep_prob: 1.0})
            print('Step: %d, Loss(ml): %f, Accuracy: %f' % (i + j * loop_len, loss_val_mlp, acc_val_mlp))

            # test_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss[j], nn.accuracy[j]],
                feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
            print('Step: %d, Loss(te): %f, Accuracy: %f' % (i + j * loop_len, loss_val, acc_val))
            nn.sess.run(nn.assign_op_total_accuracy, feed_dict={nn.input_placeholder_total_accuracy: acc_val})

            summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                [mlp.summary, mlp.loss, mlp.accuracy],
                feed_dict={mlp.x: test_data, mlp.t: test_labels, mlp.keep_prob: 1.0})
            print('Step: %d, Loss(ml): %f, Accuracy: %f' % (i + j * loop_len, loss_val_mlp, acc_val_mlp))

        nn.writer.add_summary(summary, i + j * loop_len)
        mlp.writer.add_summary(summary_mlp, i + j * loop_len)
        i += 1

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
                num_units3 = 128
                num_units2 = 128
                num_units1 = 128

        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w3 = tf.Variable(tf.truncated_normal(shape=[input_size, num_units3], mean=0.0, stddev=1 / input_size))
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
            with tf.name_scope('shortcut_weight'):
                w0_4 = tf.Variable(tf.truncated_normal(shape=[input_size, 10], mean=0.0, stddev=1 / input_size))
                w0_3 = tf.Variable(tf.truncated_normal(shape=[num_units3, 10], mean=0.0, stddev=1 / num_units3))
                w0_2 = tf.Variable(tf.truncated_normal(shape=[num_units2, 10], mean=0.0, stddev=1 / num_units2))
                gate_4 = tf.Variable(1.)
                gate_3 = tf.Variable(1.)
                gate_2 = tf.Variable(1.)
                gate_1 = tf.Variable(1.)
        with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('feed_forword'):
            with tf.name_scope('state8'):
                with tf.name_scope('hidden8_2'):
                    hidden8_3 = tf.nn.relu(tf.matmul(x, w3) + b3)
                with tf.name_scope('hidden8__drop'):
                    hidden8_3_drop = tf.nn.dropout(hidden8_3, keep_prob)

                with tf.name_scope('hidden8_2'):
                    hidden8_2 = tf.nn.relu(tf.matmul(hidden8_3_drop, w2) + b2)
                with tf.name_scope('hidden8__drop'):
                    hidden8_2_drop = tf.nn.dropout(hidden8_2, keep_prob)

                """
                with tf.name_scope('hidden8_2'):
                    hidden8_2 = tf.nn.relu(tf.matmul(x, w2) + b2)
                with tf.name_scope('hidden8__drop'):
                    hidden8_2_drop = tf.nn.dropout(hidden8_2, keep_prob)
                """
                with tf.name_scope('hidden8_1'):
                    hidden8_1 = tf.nn.relu(tf.matmul(hidden8_2_drop, w1) + b1)
                with tf.name_scope('hidden8_1_drop'):
                    hidden8_1_drop = tf.nn.dropout(hidden8_1, keep_prob)
                with tf.name_scope('p8'):
                    p8 = tf.nn.softmax(tf.matmul(x, w0_4)*tf.sigmoid(gate_4) + b0
                                       + tf.matmul(hidden8_1_drop, w0)*tf.sigmoid(gate_1)
                                       + tf.matmul(hidden8_2_drop, w0_2)*tf.sigmoid(gate_2)
                                       + tf.matmul(hidden8_3_drop, w0_3)*tf.sigmoid(gate_3))
                    """
                    p8 = tf.nn.softmax(tf.matmul(x, w0_4)*gate_4 + b0
                                       + tf.matmul(hidden8_1_drop, w0)*gate_1
                                       + tf.matmul(hidden8_2_drop, w0_2)*gate_2
                                       + tf.matmul(hidden8_3_drop, w0_3)*gate_3)
                    p8 = tf.nn.softmax(tf.matmul(x, w0_4) + b0
                                       + tf.matmul(hidden8_1_drop, w0)
                                       + tf.matmul(hidden8_2_drop, w0_2)
                                       + tf.matmul(hidden8_3_drop, w0_3))
                    """
            t = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss8'):
                loss = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p8, 1e-10, 1.0)))
            learning_rate = 0.001  # 0.0001
            # AdamOptimizer
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer().minimize(loss)
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
        self.w0_3 = w0_3
        self.w0_2 = w0_2

        self.assign_op_total_accuracy = assign_op_total_accuracy
        self.input_placeholder_total_accuracy = input_placeholder_total_accuracy

        self.assign_op_train_total_accuracy = assign_op_train_total_accuracy
        self.input_placeholder_train_total_accuracy = input_placeholder_train_total_accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/1116_shortcut_mlp_4layer", sess.graph)
        # writer = tf.summary.FileWriter("./log/1109_compare/shortcut_MLP", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer


if __name__ == '__main__':
    data = np.load("MNIST_train_data.npy")[0:50000]
    valid_data = np.load("MNIST_train_data.npy")[50000:55000]
    labels = np.load("MNIST_train_labels.npy")[0:50000]
    valid_labels = np.load("MNIST_train_labels.npy")[50000:55000]
    test_data = np.load("MNIST_test_data.npy")
    test_labels = np.load("MNIST_test_labels.npy")

    """
    data = np.load("train_data.npy")
    labels = np.load("train_label.npy")
    test_data = np.load("test_data.npy")
    test_labels = np.load("test_label.npy")

    data = np.load("cifar-10_train_data.npy")
    labels = np.load("cifar-10_train_label.npy")
    test_data = np.load("cifar-10_test_data.npy")
    test_labels = np.load("cifar-10_test_label.npy")
    """
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
    shortcut_mlp = layer()
    batchsize = 128
    batch_xs = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])
    print(data.shape)
    train_size = 50000  # 10000 is good
    train_test_size = train_size
    if train_size > 10000:
        train_test_size = 10000
    print("train_size:{0}".format(train_size))
    print("train_test_size:{0}".format(train_test_size))
    print("batch_size:{0}".format(batchsize))
    train_list_num = list(range(train_test_size))
    # train_list_num = random.sample(list(range(len(data))), train_size)

    train_test_data = np.mat([[0.0 for n in range(image_size)] for k in range(len(train_list_num))])
    train_test_labels = np.mat([[0.0 for n in range(10)] for k in range(len(train_list_num))])
    for i in range(len(train_test_data)):
        tmp = train_list_num[i]
        train_test_data[i] = data[tmp].reshape(1, image_size)
        # train_test_data[i] /= train_test_data[i].max()
        train_test_labels[i] = labels[tmp].reshape(1, 10)
        # print(train_test_labels)
        # for i in range(100):
        # print(train_test_labels[i])

    test_output_shortcut_mlp = list()
    train_test_output_shortcut_mlp = list()

    loop_len = 200000  # 400000
    print("loop_len:{0}".format(loop_len))
    drop_out_rate = 1.0
    loop_count = 0
    print("dropout:{0}".format(drop_out_rate))

    max_accuracy = 0.0
    max_accuracy_list = list()
    max_accuracy_layer_list = list()

    for i in range(loop_len):
        for n in range(batchsize):
            tmp = int(random.uniform(0, train_size))
            # batch_xs[n] = data[train_list_num[tmp]].reshape(1, image_size)
            batch_xs[n] = data[tmp].reshape(1, image_size)
            # batch_xs[n] /= batch_xs[n].max()
            batch_ts[n] = labels[tmp].reshape(1, 10)
        shortcut_mlp.sess.run(shortcut_mlp.train_step,
                              feed_dict={shortcut_mlp.x: batch_xs, shortcut_mlp.t: batch_ts,
                                         shortcut_mlp.keep_prob: drop_out_rate})
        #if i % 100 == 0:
            #print("step: {0}".format(i))
        if i % 100 == 0:
            # train_data
            summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                feed_dict={shortcut_mlp.x: train_test_data, shortcut_mlp.t: train_test_labels,
                           shortcut_mlp.keep_prob: 1.0})
            print('Step: %d, Loss(tr): %f, Accuracy: %f' % (loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))

            train_test_output_shortcut_mlp.append(acc_val_shortcut_mlp)
            train_loss = loss_val_shortcut_mlp

            # test_data
            summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                feed_dict={shortcut_mlp.x: valid_data, shortcut_mlp.t: valid_labels, shortcut_mlp.keep_prob: 1.0})
            print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))
            test_output_shortcut_mlp.append(acc_val_shortcut_mlp)

            if max_accuracy < acc_val_shortcut_mlp:
                max_accuracy = acc_val_shortcut_mlp
                print("------max_in_valid(te)------")
                summary, loss_val, acc_val = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
                print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
                max_accuracy_list.append(acc_val)
                max_accuracy_layer_list.append(i)
                print("------max_in_valid(te)_end------")
            # else:
                # print("step: {0}".format(loop_count))
        shortcut_mlp.writer.add_summary(summary_shortcut_mlp, loop_count)
        """
        if train_loss < 0.001:
            # test_data
            print("------test_data------")
            summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
            print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
                loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))
            print("------test__end------")
            break
        """
        i += 1
        loop_count += 1
    # test_data
    print("------test_data------")
    summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
        [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
        feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
    print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
        loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))
    print("------test__end------")

    print("max_test_accuracy")
    print("sh {0}:{1}".format(max(test_output_shortcut_mlp),
                              [i * 100 / loop_len for i, x in enumerate(test_output_shortcut_mlp) if
                               x == max(test_output_shortcut_mlp)]))
    print("max_train_accuracy")
    print("sh {0}:{1}".format(max(train_test_output_shortcut_mlp),
                              [i * 100 / loop_len for i, x in enumerate(train_test_output_shortcut_mlp) if
                               x >= max(train_test_output_shortcut_mlp) - 0.001]))
    print("train_size:{0}".format(train_size))
    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print(max_accuracy_list)
    print(max_accuracy_layer_list)

    print("with_sigmoid_gate")


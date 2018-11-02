import tensorflow as tf
import numpy as np
import random
import os
import xlwt
import math

# import load_notMNIST_1031

# np.random.seed(20160612)
# tf.set_random_seed(20160612)
np.random.seed(20171109)
tf.set_random_seed(20171109)

image_size = 784
output_size = 10
batchsize = 100
hidden_unit_number = 500


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
                hidden_unit = hidden_unit_number
            with tf.name_scope('num_units8'):
                num_units8 = hidden_unit
            with tf.name_scope('num_units7'):
                num_units7 = hidden_unit
            with tf.name_scope('num_units6'):
                num_units6 = hidden_unit
                num_units5 = hidden_unit
                num_units4 = hidden_unit
                num_units3 = hidden_unit
                num_units2 = hidden_unit
                num_units1 = hidden_unit
                num_units0 = output_size

        with tf.name_scope('layer8'):
            with tf.name_scope('w8'):
                """
                w8 = tf.Variable(
                    tf.random_uniform(shape=[input_size, num_units8],
                                      minval=-np.sqrt(6.0 / input_size),
                                      maxval=np.sqrt(6.0 / input_size),
                                      dtype=tf.float32))
                """
                w8 = tf.Variable(
                    tf.random_uniform(shape=[input_size, num_units8],
                                      minval=-np.sqrt(6.0 / (input_size + num_units8)),
                                      maxval=np.sqrt(6.0 / (input_size + num_units8)),
                                      dtype=tf.float32))
            with tf.name_scope('b8'):
                b8 = tf.Variable(tf.constant(0.1, shape=[num_units8]))
        with tf.name_scope('layer7'):
            with tf.name_scope('w7'):
                w7 = tf.Variable(
                    tf.random_uniform(shape=[num_units8, num_units7],
                                      minval=-np.sqrt(6.0 / (num_units8 + num_units7)),
                                      maxval=np.sqrt(6.0 / (num_units8 + num_units7)),
                                      dtype=tf.float32))
            with tf.name_scope('b7'):
                b7 = tf.Variable(tf.constant(0.1, shape=[num_units7]))
        with tf.name_scope('layer6'):
            with tf.name_scope('w6'):
                w6 = tf.Variable(
                    tf.random_uniform(shape=[num_units7, num_units6],
                                      minval=-np.sqrt(6.0 / (num_units7 + num_units6)),
                                      maxval=np.sqrt(6.0 / (num_units7 + num_units6)),
                                      dtype=tf.float32))
            with tf.name_scope('b6'):
                b6 = tf.Variable(tf.constant(0.1, shape=[num_units6]))
        with tf.name_scope('layer5'):
            with tf.name_scope('w5'):
                w5 = tf.Variable(
                    tf.random_uniform(shape=[num_units6, num_units5],
                                      minval=-np.sqrt(6.0 / (num_units6 + num_units5)),
                                      maxval=np.sqrt(6.0 / (num_units6 + num_units5)),
                                      dtype=tf.float32))
            with tf.name_scope('b5'):
                b5 = tf.Variable(tf.constant(0.1, shape=[num_units5]))
        with tf.name_scope('layer4'):
            with tf.name_scope('w4'):
                w4 = tf.Variable(
                    tf.random_uniform(shape=[num_units5, num_units4],
                                      minval=-np.sqrt(6.0 / (num_units5 + num_units4)),
                                      maxval=np.sqrt(6.0 / (num_units5 + num_units4)),
                                      dtype=tf.float32))
            with tf.name_scope('b4'):
                b4 = tf.Variable(tf.constant(0.1, shape=[num_units4]))
        with tf.name_scope('layer3'):
            with tf.name_scope('w3'):
                w3 = tf.Variable(
                    tf.random_uniform(shape=[num_units4, num_units3],
                                      minval=-np.sqrt(6.0 / (num_units4 + num_units3)),
                                      maxval=np.sqrt(6.0 / (num_units4 + num_units3)),
                                      dtype=tf.float32))
            with tf.name_scope('b3'):
                b3 = tf.Variable(tf.constant(0.1, shape=[num_units3]))
        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w2 = tf.Variable(
                    tf.random_uniform(shape=[num_units3, num_units2],
                                      minval=-np.sqrt(6.0 / (num_units3 + num_units2)),
                                      maxval=np.sqrt(6.0 / (num_units3 + num_units2)),
                                      dtype=tf.float32))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
        with tf.name_scope('layer1'):
            with tf.name_scope('w1'):
                w1 = tf.Variable(
                    tf.random_uniform(shape=[num_units2, num_units1],
                                      minval=-np.sqrt(6.0 / (num_units2 + num_units1)),
                                      maxval=np.sqrt(6.0 / (num_units2 + num_units1)),
                                      dtype=tf.float32))
            with tf.name_scope('b1'):
                b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))
        with tf.name_scope('layer0'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(
                    tf.random_uniform(shape=[num_units1, num_units0],
                                      minval=-np.sqrt(6.0 / (num_units1 + num_units0)),
                                      maxval=np.sqrt(6.0 / (num_units1 + num_units0)),
                                      dtype=tf.float32))

        with tf.name_scope('layer0'):
            with tf.name_scope('w0_9'):
                w0_9 = tf.Variable(
                    tf.random_uniform(shape=[input_size, output_size],
                                      minval=-np.sqrt(6.0 / (input_size + output_size)),
                                      maxval=np.sqrt(6.0 / (input_size + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w7_9'):
                w0_8 = tf.Variable(
                    tf.random_uniform(shape=[num_units8, output_size],
                                      minval=-np.sqrt(6.0 / (num_units8 + output_size)),
                                      maxval=np.sqrt(6.0 / (num_units8 + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w7_9'):
                w0_7 = tf.Variable(
                    tf.random_uniform(shape=[num_units7, output_size],
                                      minval=-np.sqrt(6.0 / (num_units7 + output_size)),
                                      maxval=np.sqrt(6.0 / (num_units7 + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w6_9'):
                w0_6 = tf.Variable(
                    tf.random_uniform(shape=[num_units6, output_size],
                                      minval=-np.sqrt(6.0 / (num_units6 + output_size)),
                                      maxval=np.sqrt(6.0 / (num_units6 + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([output_size]))

                """
                gate_9 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_8 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_7 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_6 = tf.Variable(tf.constant(1.0, shape=[1]))
                """
        with tf.name_scope('feed_forword'):
            hidden8 = tf.nn.relu(tf.matmul(x, w8) + b8)
            hidden8_drop = tf.nn.dropout(hidden8, keep_prob)

            hidden7 = tf.nn.relu(tf.matmul(hidden8_drop, w7) + b7)
            hidden7_drop = tf.nn.dropout(hidden7, keep_prob)

            hidden6 = tf.nn.relu(tf.matmul(hidden7_drop, w6) + b6)
            hidden6_drop = tf.nn.dropout(hidden6, keep_prob)

            hidden5 = tf.nn.relu(tf.matmul(hidden6_drop, w5) + b5)
            hidden5_drop = tf.nn.dropout(hidden5, keep_prob)

            hidden4 = tf.nn.relu(tf.matmul(hidden5_drop, w4) + b4)
            hidden4_drop = tf.nn.dropout(hidden4, keep_prob)

            hidden3 = tf.nn.relu(tf.matmul(hidden4_drop, w3) + b3)
            hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

            hidden2 = tf.nn.relu(tf.matmul(hidden3_drop, w2) + b2)
            hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

            hidden1 = tf.nn.relu(tf.matmul(hidden2_drop, w1) + b1)
            hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

            with tf.name_scope('p0'):
                p0 = tf.nn.softmax(tf.matmul(x, w0_9) + b0)
            with tf.name_scope('p1'):
                p1 = tf.nn.softmax(tf.matmul(hidden8_drop, w0_8) + b0)
            with tf.name_scope('p2'):
                p2 = tf.nn.softmax(tf.matmul(hidden7_drop, w0_7) + b0)
            with tf.name_scope('p3'):
                p3 = tf.nn.softmax(b0 + tf.matmul(hidden6_drop, w0_6))
            with tf.name_scope('p3'):
                p4 = tf.nn.softmax(b0 + tf.matmul(hidden1_drop, w0))

            t = tf.placeholder(tf.float32, [None, output_size])

        with tf.name_scope('optimizer'):
            with tf.name_scope('loss0'):
                loss0 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0))) / batchsize
            with tf.name_scope('loss1'):
                loss1 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0))) / batchsize
            with tf.name_scope('loss2'):
                loss2 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p2, 1e-10, 1.0))) / batchsize
            with tf.name_scope('loss3'):
                loss3 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p3, 1e-10, 1.0))) / batchsize
            with tf.name_scope('loss4'):
                loss4 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p4, 1e-10, 1.0))) / batchsize
            loss = [loss0, loss1, loss2, loss3, loss4]
            # Adam
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer().minimize(loss0)
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer().minimize(loss1)
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer().minimize(loss2)
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer().minimize(loss3)
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer().minimize(loss4)
            # Adam
            """
            with tf.name_scope('train_step0'):
                train_step0 = tf.train.AdamOptimizer().minimize(loss0, var_list=[b0, w0_9])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer().minimize(loss1, var_list=[b0, w0_8, w8, b8])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer().minimize(loss2, var_list=[b0, w0_7, w7, b7])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer().minimize(loss3, var_list=[b0, w0_6, w6, b6])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer().minimize(loss4)
            """
            train_step = [train_step0, train_step1, train_step2, train_step3, train_step4]
            with tf.name_scope('correct_prediction'):
                correct_prediction0 = tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1))
                correct_prediction1 = tf.equal(tf.argmax(p1, 1), tf.argmax(t, 1))
                correct_prediction2 = tf.equal(tf.argmax(p2, 1), tf.argmax(t, 1))
                correct_prediction3 = tf.equal(tf.argmax(p3, 1), tf.argmax(t, 1))
                correct_prediction4 = tf.equal(tf.argmax(p4, 1), tf.argmax(t, 1))
            with tf.name_scope('accuracy'):
                accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
                accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
                accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
                accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
                accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
                accuracy = [accuracy0, accuracy1, accuracy2, accuracy3, accuracy4]

        with tf.name_scope('assign_input'):
            total_accuracy = tf.Variable(0.)
            input_placeholder_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_total_accuracy = total_accuracy.assign(input_placeholder_total_accuracy)

            train_total_accuracy = tf.Variable(0.)
            input_placeholder_train_total_accuracy = tf.placeholder(tf.float32, shape=[])
            assign_op_train_total_accuracy = train_total_accuracy.assign(input_placeholder_train_total_accuracy)

            total_loss = tf.Variable(0.)
            input_placeholder_total_loss = tf.placeholder(tf.float32, shape=[])
            assign_op_total_loss = total_loss.assign(input_placeholder_total_loss)

            train_total_loss = tf.Variable(0.)
            input_placeholder_train_total_loss = tf.placeholder(tf.float32, shape=[])
            assign_op_train_total_loss = train_total_loss.assign(input_placeholder_train_total_loss)

        tf.summary.scalar("accuracy0", tf.reduce_sum(accuracy[0]))
        tf.summary.scalar("accuracy1", tf.reduce_sum(accuracy[1]))
        tf.summary.scalar("accuracy2", tf.reduce_sum(accuracy[2]))
        tf.summary.scalar("accuracy3", tf.reduce_sum(accuracy[3]))
        tf.summary.scalar("accuracy4", tf.reduce_sum(accuracy[4]))
        tf.summary.scalar("w6", tf.reduce_sum(w6))
        tf.summary.scalar("w7", tf.reduce_sum(w7))
        tf.summary.scalar("w8", tf.reduce_sum(w8))

        tf.summary.scalar("w0_9", tf.reduce_sum(w0_9))
        tf.summary.scalar("w0_8", tf.reduce_sum(w0_8))
        tf.summary.scalar("w0_7", tf.reduce_sum(w0_7))
        tf.summary.scalar("w0_6", tf.reduce_sum(w0_6))
        tf.summary.scalar("total_accuracy", tf.reduce_sum(total_accuracy))
        tf.summary.scalar("train_total_accuracy", tf.reduce_sum(train_total_accuracy))
        tf.summary.scalar("total_loss", tf.reduce_sum(total_loss))
        tf.summary.scalar("train_total_loss", tf.reduce_sum(train_total_loss))
        tf.summary.scalar("accuracy", tf.reduce_sum(total_accuracy))
        """
        tf.summary.scalar("loss0", loss0)
        tf.summary.scalar("loss1", loss1)
        tf.summary.scalar("loss2", loss2)
        tf.summary.scalar("loss3", loss3)
        tf.summary.scalar("loss4", loss4)
        tf.summary.scalar("loss5", loss5)
        tf.summary.scalar("loss6", loss6)
        tf.summary.scalar("loss7", loss7)
        tf.summary.scalar("loss8", loss8)
        """
        self.x, self.t = x, t
        self.loss = loss
        self.accuracy = accuracy
        self.total_accuracy = total_accuracy
        self.total_loss = total_loss
        self.train_total_loss = train_total_loss
        self.train_step = train_step
        self.keep_prob = keep_prob
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.w0_9 = w0_9
        self.w0_8 = w0_8
        self.w0_7 = w0_7
        self.w0_6 = w0_6

        self.assign_op_total_accuracy = assign_op_total_accuracy
        self.input_placeholder_total_accuracy = input_placeholder_total_accuracy

        self.assign_op_train_total_accuracy = assign_op_train_total_accuracy
        self.input_placeholder_train_total_accuracy = input_placeholder_train_total_accuracy

        self.assign_op_train_total_loss = assign_op_train_total_loss
        self.input_placeholder_train_total_loss = input_placeholder_train_total_loss

        self.assign_op_total_loss = assign_op_total_loss
        self.input_placeholder_total_loss = input_placeholder_total_loss

        """
        self.gate_9 = gate_9
        self.gate_8 = gate_8
        self.gate_7 = gate_7
        self.gate_6 = gate_6
        """
        self.hidden_unit = hidden_unit

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./log/1105_MNIST_nonliner", sess.graph)
        writer = tf.summary.FileWriter("./log/1203_MNIST_MLP_5layer_epoch", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = tf.train.Saver(max_to_keep=None)


def initialize():
    global batchsize
    batchsize = 10
    global image_size
    image_size = 3072#784
    global hidden_unit_number
    hidden_unit_number = 500


if __name__ == '__main__':
    """
    data = np.load("cifar10_train_data.npy")
    labels = np.load("cifar10_train_labels.npy")
    test_data = np.load("cifar10_test_data.npy")
    test_labels = np.load("cifar10_test_labels.npy")

    data = np.load("iris_train_data.npy")
    labels = np.load("iris_train_labels.npy")
    test_data = np.load("iris_test_data.npy")
    test_labels = np.load("iris_test_labels.npy")
    """
    initialize()

    nn = layer()
    # MLP = __import__("1116_MNIST_MLP_4layer")
    # mlp = MLP.layer()
    # shortcut_MLP = __import__("1116_shortcut_mlp_4layer")
    # shortcut_mlp = shortcut_MLP.layer()
    train_size = 1000  # 10000 is good
    layer_num = 4
    early_stopping_num = 200
    session_name = "./session_log/saver_1203_MNIST_MLP_5layer_epoch_5"
    batch_xs = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(output_size)] for k in range(batchsize)])

    """
    data = np.load("MNIST_train_data.npy")[0:train_size]
    labels = np.load("MNIST_train_labels.npy")[0:train_size]
    test_data = np.load("MNIST_test_data.npy")
    test_labels = np.load("MNIST_test_labels.npy")
    valid_data = np.load("MNIST_valid_data.npy")[0:int(train_size/10)]
    valid_labels = np.load("MNIST_valid_labels.npy")[0:int(train_size/10)]
    """
    """
    data = np.load("fashion_train_data.npy")[0:train_size]
    labels = np.load("fashion_train_labels.npy")[0:train_size]
    test_data = np.load("fashion_test_data.npy")
    test_labels = np.load("fashion_test_labels.npy")
    valid_data = np.load("fashion_valid_data.npy")[0:int(train_size / 10)]
    valid_labels = np.load("fashion_valid_labels.npy")[0:int(train_size / 10)]
    """
    data = np.load("cifar-10_train_data_normalized.npy")[0:train_size]
    labels = np.load("cifar-10_train_labels.npy")[0:train_size]
    test_data = np.load("cifar-10_test_data_normalized.npy")
    test_labels = np.load("cifar-10_test_labels.npy")
    valid_data = np.load("cifar-10_train_data_normalized.npy")[46000:46000 + int(train_size / 10)]
    valid_labels = np.load("cifar-10_train_labels.npy")[46000:46000 + int(train_size / 10)]
    print(data.shape)

    train_test_size = train_size
    if train_size > 20000:
        train_test_size = 20000
    print("train_size:{0}".format(train_size))
    print("train_test_size:{0}".format(train_test_size))
    print("batch_size:{0}".format(batchsize))
    train_list_num = list(range(train_test_size))
    # train_list_num = random.sample(list(range(len(data))), train_size)

    train_test_data = np.mat([[0.0 for n in range(image_size)] for k in range(len(train_list_num))])
    train_test_labels = np.mat([[0.0 for n in range(output_size)] for k in range(len(train_list_num))])
    for i in range(len(train_test_data)):
        tmp = train_list_num[i]
        train_test_data[i] = data[tmp].reshape(1, image_size)
        # train_test_data[i] /= train_test_data[i].max()
        train_test_labels[i] = labels[tmp].reshape(1, output_size)

    test_output = list()
    train_test_output = list()
    # test_output_mlp = list()
    # train_test_output_mlp = list()
    # test_output_shortcut_mlp = list()
    # train_test_output_shortcut_mlp = list()
    max_accuracy = 0.0
    max_accuracy_list = list()
    max_accuracy_layer_list = list()

    loop_len = 100000  # 400000
    print("loop_len:{0}".format(loop_len))
    drop_out_rate = 1.0
    loop_count = 0
    acc_val = 0.
    gate_value = [1., 1., 1., 1.]
    print("dropout:{0}".format(drop_out_rate))
    train_accracy = 0.0
    best_loss = 10000.

    for j in range(layer_num, layer_num + 1):
        print("step:{0} start".format(j))
        stopping_step = 0
        best_loss = 10000.
        for i in range(loop_len):
            each_epoch = list(range(train_size))
            random.shuffle(each_epoch)
            while len(each_epoch) >= batchsize:
                for n in range(batchsize):
                    tmp = each_epoch[0]
                    each_epoch.remove(tmp)
                    batch_xs[n] = data[tmp].reshape(1, image_size)
                    batch_ts[n] = labels[tmp].reshape(1, output_size)
                nn.sess.run(nn.train_step[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate})

            tmp_loss = nn.sess.run(nn.loss[j], feed_dict={nn.x: valid_data, nn.t: valid_labels,
                                                          nn.keep_prob: drop_out_rate})
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                stopping_step = 0
                nn.saver.save(nn.sess, session_name)
            else:
                stopping_step += 1
            if stopping_step >= early_stopping_num:
                nn.writer.add_summary(summary, loop_count)
                print("early_stopping is trigger at step:{0}".format(loop_count - early_stopping_num))
                loop_count -= early_stopping_num
                nn.saver.restore(nn.sess, session_name)
                best_loss = 10000.0
                break

            if i % 1 == 0:
                # train_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
                print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
                train_loss = loss_val
                train_accracy = acc_val
                nn.sess.run(nn.assign_op_train_total_loss, feed_dict={nn.input_placeholder_train_total_loss: loss_val})

                train_test_output.append(acc_val)

                # valid_data
                loss_val, acc_val = nn.sess.run(
                    [nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0})
                print('Step(%d): %d, Loss(va): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))

                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
                print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                test_output.append(acc_val)
            nn.writer.add_summary(summary, loop_count)
            i += 1
            loop_count += 1
        # test_data
        print("------test_data------")
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0})
        print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
        print("------test__end------")
        # gate_value[j] = train_accracy
        j += 1
    print("max_test_accuracy")
    print("te {0}:{1}".format(max(test_output),
                              [i * 100 for i, x in enumerate(test_output) if x == max(test_output)]))
    print("max_train_accuracy")
    print("tr {0}:{1}".format(max(train_test_output), [i * 100 for i, x in enumerate(train_test_output) if
                                                       x == max(train_test_output)]))
    print("train_size:{0}".format(train_size))

    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print("dropout:{0}".format(drop_out_rate))
    print("hidden_nodes:{0}".format(nn.hidden_unit))

    # test_data
    print("train_size:{0}".format(train_size))
    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print("dropout:{0}".format(drop_out_rate))
    print("hidden_nodes:{0}".format(nn.hidden_unit))
    print("valid_size:{0}".format(len(valid_data)))
    print("early_stopping_num:{0}".format(early_stopping_num))
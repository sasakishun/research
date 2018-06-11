import tensorflow as tf
import numpy as np
import random
import os
import xlwt

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
                hidden_unit = 100
            with tf.name_scope('num_units8'):
                num_units8 = hidden_unit
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
                # b0 = tf.Variable(tf.constant(0.1, shape=[10]))
                b0 = tf.Variable(tf.zeros([10]))
                b0_9 = tf.Variable(tf.constant(0.1, shape=[10]))
                b0_8 = tf.Variable(tf.constant(0.1, shape=[10]))
                b0_7 = tf.Variable(tf.constant(0.1, shape=[10]))
                b0_6 = tf.Variable(tf.constant(0.1, shape=[10]))

                gate_9 = tf.placeholder(tf.float32)
                gate_8 = tf.placeholder(tf.float32)
                gate_7 = tf.placeholder(tf.float32)
                gate_6 = tf.placeholder(tf.float32)

                # gate_9 = tf.Variable(1.0)
                # gate_8 = tf.Variable(1.0)
                # gate_7 = tf.Variable(1.0)
                # gate_6 = tf.Variable(1.0)

                # gate_9 = tf.Variable(tf.constant(0.1, shape=[10]))
                # gate_8 = tf.Variable(tf.constant(0.1, shape=[10]))
                # gate_7 = tf.Variable(tf.constant(0.1, shape=[10]))
                # gate_6 = tf.Variable(tf.constant(0.1, shape=[10]))
                # gate_5 = tf.Variable(1.0)
                # gate_4 = tf.Variable(1.0)
                # gate_3 = tf.Variable(1.0)
                # gate_2 = tf.Variable(1.0)
        with tf.name_scope('feed_forword'):
            with tf.name_scope('state0'):
                with tf.name_scope('p0'):
                    p0 = tf.nn.softmax(tf.matmul(x, w0_9) * gate_9 + b0)

            with tf.name_scope('state1'):
                with tf.name_scope('hidden1_1'):
                    hidden1_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden1_1_drop'):
                    hidden1_8_drop = tf.nn.dropout(hidden1_8, keep_prob)
                with tf.name_scope('p1'):
                    p1 = tf.nn.softmax(tf.matmul(hidden1_8_drop, w0_8) * gate_8
                                       + b0
                                       + tf.matmul(x, w0_9) * gate_9)

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
                    p2 = tf.nn.softmax(tf.matmul(hidden2_7_drop, w0_7) * gate_7
                                       + b0
                                       + tf.matmul(x, w0_9) * gate_9
                                       + tf.matmul(hidden2_8, w0_8) * gate_8)

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
                    p3 = tf.nn.softmax(tf.matmul(x, w0_9) * gate_9
                                       + b0
                                       + tf.matmul(hidden3_6_drop, w0_6) * gate_6
                                       + tf.matmul(hidden3_7_drop, w0_7) * gate_7
                                       + tf.matmul(hidden3_8_drop, w0_8) * gate_8)

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

            with tf.name_scope('state6'):
                with tf.name_scope('hidden6_6'):
                    hidden6_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden6_6_drop'):
                    hidden6_8_drop = tf.nn.dropout(hidden6_8, keep_prob)

                with tf.name_scope('hidden6_5'):
                    hidden6_7 = tf.nn.relu(tf.matmul(hidden6_8_drop, w7) + b7)
                with tf.name_scope('hidden6__drop'):
                    hidden6_7_drop = tf.nn.dropout(hidden6_7, keep_prob)

                with tf.name_scope('hidden6_4'):
                    hidden6_6 = tf.nn.relu(tf.matmul(hidden6_7_drop, w6) + b6)
                with tf.name_scope('hidden6__drop'):
                    hidden6_6_drop = tf.nn.dropout(hidden6_6, keep_prob)

                with tf.name_scope('hidden6_3'):
                    hidden6_5 = tf.nn.relu(tf.matmul(hidden6_6_drop, w5) + b5)
                with tf.name_scope('hidden6__drop'):
                    hidden6_5_drop = tf.nn.dropout(hidden6_5, keep_prob)

                with tf.name_scope('hidden6_2'):
                    hidden6_4 = tf.nn.relu(tf.matmul(hidden6_5_drop, w4) + b4)
                with tf.name_scope('hidden6_2_drop'):
                    hidden6_4_drop = tf.nn.dropout(hidden6_4, keep_prob)

                with tf.name_scope('hidden6_1'):
                    hidden6_3 = tf.nn.relu(tf.matmul(hidden6_4_drop, w3) + b3)
                with tf.name_scope('hidden6_1_drop'):
                    hidden6_3_drop = tf.nn.dropout(hidden6_3, keep_prob)
                with tf.name_scope('p6'):
                    p6 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden6_3_drop, w0_3)
                                       + tf.matmul(hidden6_4_drop, w0_4)
                                       + tf.matmul(hidden6_5_drop, w0_5)
                                       + tf.matmul(hidden6_6_drop, w0_6)
                                       + tf.matmul(hidden6_7_drop, w0_7)
                                       + tf.matmul(hidden6_8_drop, w0_8))

            with tf.name_scope('state7'):
                with tf.name_scope('hidden7_7'):
                    hidden7_8 = tf.nn.relu(tf.matmul(x, w8) + b8)
                with tf.name_scope('hidden7_7_drop'):
                    hidden7_8_drop = tf.nn.dropout(hidden7_8, keep_prob)

                with tf.name_scope('hidden7_6'):
                    hidden7_7 = tf.nn.relu(tf.matmul(hidden7_8_drop, w7) + b7)
                with tf.name_scope('hidden7_6_drop'):
                    hidden7_7_drop = tf.nn.dropout(hidden7_7, keep_prob)

                with tf.name_scope('hidden7_5'):
                    hidden7_6 = tf.nn.relu(tf.matmul(hidden7_7_drop, w6) + b6)
                with tf.name_scope('hidden8__drop'):
                    hidden7_6_drop = tf.nn.dropout(hidden7_6, keep_prob)

                with tf.name_scope('hidden7_4'):
                    hidden7_5 = tf.nn.relu(tf.matmul(hidden7_6_drop, w5) + b5)
                with tf.name_scope('hidden8__drop'):
                    hidden7_5_drop = tf.nn.dropout(hidden7_5, keep_prob)

                with tf.name_scope('hidden7_3'):
                    hidden7_4 = tf.nn.relu(tf.matmul(hidden7_5_drop, w4) + b4)
                with tf.name_scope('hidden7__drop'):
                    hidden7_4_drop = tf.nn.dropout(hidden7_4, keep_prob)

                with tf.name_scope('hidden7_2'):
                    hidden7_3 = tf.nn.relu(tf.matmul(hidden7_4_drop, w3) + b3)
                with tf.name_scope('hidden7_2_drop'):
                    hidden7_3_drop = tf.nn.dropout(hidden7_3, keep_prob)

                with tf.name_scope('hidden7_1'):
                    hidden7_2 = tf.nn.relu(tf.matmul(hidden7_3_drop, w2) + b2)
                with tf.name_scope('hidden7_1_drop'):
                    hidden7_2_drop = tf.nn.dropout(hidden7_2, keep_prob)
                with tf.name_scope('p7'):
                    p7 = tf.nn.softmax(tf.matmul(x, w0_9) + b0
                                       + tf.matmul(hidden7_2_drop, w0_2)
                                       + tf.matmul(hidden7_3_drop, w0_3)
                                       + tf.matmul(hidden7_4_drop, w0_4)
                                       + tf.matmul(hidden7_5_drop, w0_5)
                                       + tf.matmul(hidden7_6_drop, w0_6)
                                       + tf.matmul(hidden7_7_drop, w0_7)
                                       + tf.matmul(hidden7_8_drop, w0_8))

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
            with tf.name_scope('loss6'):
                loss6 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p6, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_5, 1e-10, 1.0)))
            with tf.name_scope('loss7'):
                loss7 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p7, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_4, 1e-10, 1.0)))
            with tf.name_scope('loss8'):
                loss8 = -tf.reduce_sum(t * tf.log(
                    tf.clip_by_value(p8, 1e-10, 1.0)))  # - tf.reduce_sum(tf.log(tf.clip_by_value(w0_3, 1e-10, 1.0)))
            loss = [loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8]
            # learning_rate = 0.001  # 0.0001
            # AdamOptimizer
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
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdamOptimizer().minimize(loss5)
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.AdamOptimizer().minimize(loss6)
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.AdamOptimizer().minimize(loss7)
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.AdamOptimizer().minimize(loss8)
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
                train_step0 = tf.train.AdamOptimizer().minimize(loss0, var_list=[b0, w0_9])
            with tf.name_scope('train_step1'):
                train_step1 = tf.train.AdamOptimizer().minimize(loss1, var_list=[b0, w0_8, w8, b8])
            with tf.name_scope('train_step2'):
                train_step2 = tf.train.AdamOptimizer().minimize(loss2, var_list=[b0, w0_7, w7, b7])
            with tf.name_scope('train_step3'):
                train_step3 = tf.train.AdamOptimizer().minimize(loss3, var_list=[b0, w0_6, w6, b6])
            with tf.name_scope('train_step4'):
                train_step4 = tf.train.AdamOptimizer().minimize(loss4, var_list=[b0, w0_5, w5, b5])
            with tf.name_scope('train_step5'):
                train_step5 = tf.train.AdamOptimizer().minimize(loss5, var_list=[b0, w0_4, w4, b4])
            with tf.name_scope('train_step6'):
                train_step6 = tf.train.AdamOptimizer().minimize(loss6, var_list=[b0, w0_3, w3, b3])
            with tf.name_scope('train_step7'):
                train_step7 = tf.train.AdamOptimizer().minimize(loss7, var_list=[b0, w0_2, w2, b2])
            with tf.name_scope('train_step8'):
                train_step8 = tf.train.AdamOptimizer().minimize(loss8, var_list=[b0, w0, w1, b1])
            """
            train_step = [train_step0, train_step1, train_step2, train_step3, train_step4, train_step5, train_step6,
                          train_step7, train_step8]
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
                accuracy6 = tf.reduce_mean(tf.cast(correct_prediction6, tf.float32))
                accuracy7 = tf.reduce_mean(tf.cast(correct_prediction7, tf.float32))
                accuracy8 = tf.reduce_mean(tf.cast(correct_prediction8, tf.float32))
                accuracy = [accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7,
                            accuracy8]

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
        tf.summary.scalar("accuracy5", tf.reduce_sum(accuracy[5]))
        tf.summary.scalar("accuracy6", tf.reduce_sum(accuracy[6]))
        tf.summary.scalar("accuracy7", tf.reduce_sum(accuracy[7]))
        tf.summary.scalar("accuracy8", tf.reduce_sum(accuracy[8]))
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

        self.assign_op_train_total_loss = assign_op_train_total_loss
        self.input_placeholder_train_total_loss = input_placeholder_train_total_loss

        self.assign_op_total_loss = assign_op_total_loss
        self.input_placeholder_total_loss = input_placeholder_total_loss

        self.gate_9 = gate_9
        self.gate_8 = gate_8
        self.gate_7 = gate_7
        self.gate_6 = gate_6
        self.hidden_unit = hidden_unit

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./log/1105_MNIST_nonliner", sess.graph)
        writer = tf.summary.FileWriter("./log/1120_MNIST_nonliner", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = tf.train.Saver(max_to_keep=None)


if __name__ == '__main__':
    data = np.load("MNIST_train_data.npy")[0:55000]
    valid_data = np.load("MNIST_train_data.npy")[50000:55000]
    labels = np.load("MNIST_train_labels.npy")[0:55000]
    valid_labels = np.load("MNIST_train_labels.npy")[50000:55000]
    test_data = np.load("MNIST_test_data.npy")
    test_labels = np.load("MNIST_test_labels.npy")

    """
    data = np.load("cifar10_train_data.npy")
    labels = np.load("cifar10_train_labels.npy")
    test_data = np.load("cifar10_test_data.npy")
    test_labels = np.load("cifar10_test_labels.npy")
    """
    nn = layer()
    # MLP = __import__("1116_MNIST_MLP_4layer")
    # mlp = MLP.layer()
    # shortcut_MLP = __import__("1116_shortcut_mlp_4layer")
    # shortcut_mlp = shortcut_MLP.layer()
    batchsize = 100
    batch_xs = np.mat([[0.0 for n in range(image_size)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(10)] for k in range(batchsize)])
    print(data.shape)
    train_size = 55000  # 10000 is good
    train_test_size = train_size
    if train_size > 20000:
        train_test_size = 20000
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
    # decay_rate = 0.999999
    # print("decay_rate:{0}".format(decay_rate))
    drop_out_rate = 1.0
    loop_count = 0
    acc_val = 0.
    gate_value = [1., 1., 1., 1.]
    print("dropout:{0}".format(drop_out_rate))
    train_accracy = 0.0
    best_loss = 10000.
    stopping_step = 0
    early_stopping_num = 1000
    session_name = "saver_1122_MNIST_nonliner_55000_4"
    # for j in range(len(nn.train_step)):
    for j in range(4):
        print("step:{0} start".format(j))
        # stopping_step = 0
        for i in range(loop_len):
            for n in range(batchsize):
                tmp = int(random.uniform(0, train_size))
                """
                if i + j * loop_len == 100000:
                    tmp = int(10)
                """
                # batch_xs[n] = data[train_list_num[tmp]].reshape(1, image_size)
                batch_xs[n] = data[tmp].reshape(1, image_size)
                # batch_xs[n] /= batch_xs[n].max()
                batch_ts[n] = labels[tmp].reshape(1, 10)
            nn.sess.run(nn.train_step[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                                     nn.gate_9: gate_value[0],
                                                     nn.gate_8: gate_value[1],
                                                     nn.gate_7: gate_value[2],
                                                     nn.gate_6: gate_value[3]})
            """
            mlp.sess.run(mlp.train_step, feed_dict={mlp.x: batch_xs, mlp.t: batch_ts, mlp.keep_prob: drop_out_rate})
            shortcut_mlp.sess.run(shortcut_mlp.train_step,
                                  feed_dict={shortcut_mlp.x: batch_xs, shortcut_mlp.t: batch_ts,
                                             shortcut_mlp.keep_prob: drop_out_rate})
            """
            """
            if i % 1 == 0:
                # train_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0})
                print('Step: %d, Loss(tr): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
                train_loss = loss_val
                nn.sess.run(nn.assign_op_train_total_loss, feed_dict={nn.input_placeholder_train_total_loss: loss_val})

                summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                    [mlp.summary, mlp.loss, mlp.accuracy],
                    feed_dict={mlp.x: train_test_data, mlp.t: train_test_labels, mlp.keep_prob: 1.0})
                print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

                summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: train_test_data, shortcut_mlp.t: train_test_labels,
                               shortcut_mlp.keep_prob: 1.0})
                print(
                    'Step: %d, Loss(sh): %f, Accuracy: %f' % (loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))

                train_test_output.append(acc_val)
                train_test_output_mlp.append(acc_val_mlp)
                train_test_output_shortcut_mlp.append(acc_val_shortcut_mlp)

                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
                print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_total_accuracy, feed_dict={nn.input_placeholder_total_accuracy: acc_val})
                nn.sess.run(nn.assign_op_total_loss, feed_dict={nn.input_placeholder_total_loss: loss_val})

                summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                    [mlp.summary, mlp.loss, mlp.accuracy],
                    feed_dict={mlp.x: test_data, mlp.t: test_labels, mlp.keep_prob: 1.0})
                print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

                summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
                print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
                    loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))

                test_output.append(acc_val)
                test_output_mlp.append(acc_val_mlp)
                test_output_shortcut_mlp.append(acc_val_shortcut_mlp)
            """
            tmp_loss = nn.sess.run(nn.loss[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                                          nn.gate_9: gate_value[0],
                                                          nn.gate_8: gate_value[1],
                                                          nn.gate_7: gate_value[2],
                                                          nn.gate_6: gate_value[3]})
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                stopping_step = 0
                nn.saver.save(nn.sess, session_name)
            else:
                stopping_step += 1
            if stopping_step > early_stopping_num:
                nn.writer.add_summary(summary, loop_count)
                print("early_stopping is trigger at step:{0}".format(loop_count-early_stopping_num))
                loop_count -= early_stopping_num
                nn.saver.restore(nn.sess, session_name)
                best_loss = 10000.0
                break

            if i % 100 == 0:
                # train_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0,
                               nn.gate_9: gate_value[0],
                               nn.gate_8: gate_value[1],
                               nn.gate_7: gate_value[2],
                               nn.gate_6: gate_value[3]})
                print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val})
                train_loss = loss_val
                train_accracy = acc_val
                nn.sess.run(nn.assign_op_train_total_loss, feed_dict={nn.input_placeholder_train_total_loss: loss_val})

                """
                summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                    [mlp.summary, mlp.loss, mlp.accuracy],
                    feed_dict={mlp.x: train_test_data, mlp.t: train_test_labels, mlp.keep_prob: 1.0})
                print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

                summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: train_test_data, shortcut_mlp.t: train_test_labels,
                               shortcut_mlp.keep_prob: 1.0})
                print(
                    'Step: %d, Loss(sh): %f, Accuracy: %f' % (loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))

                """
                train_test_output.append(acc_val)
                # train_test_output_mlp.append(acc_val_mlp)
                # train_test_output_shortcut_mlp.append(acc_val_shortcut_mlp)
                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0,
                               nn.gate_9: gate_value[0],
                               nn.gate_8: gate_value[1],
                               nn.gate_7: gate_value[2],
                               nn.gate_6: gate_value[3]})
                print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                # nn.sess.run(nn.assign_op_total_accuracy, feed_dict={nn.input_placeholder_total_accuracy: acc_val})
                # nn.sess.run(nn.assign_op_total_loss, feed_dict={nn.input_placeholder_total_loss: loss_val})
                """
                if max_accuracy < acc_val:
                    max_accuracy = acc_val
                    print("------max_in_valid(te)------")
                    summary, loss_val, acc_val = nn.sess.run(
                        [nn.summary, nn.loss[j], nn.accuracy[j]],
                        feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0,
                                   nn.gate_9: 1.0, nn.gate_8: 1.0, nn.gate_7: 1.0, nn.gate_6: 1.0})
                    print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
                    max_accuracy_list.append(acc_val)
                    max_accuracy_layer_list.append(j)
                    print("------max_in_valid(te)_end------")
                    # else:
                    # print("step: {0}".format(loop_count))
                """
                """
                summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                    [mlp.summary, mlp.loss, mlp.accuracy],
                    feed_dict={mlp.x: valid_data, mlp.t: valid_labels, mlp.keep_prob: 1.0})
                print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

                summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: valid_data, shortcut_mlp.t: valid_labels, shortcut_mlp.keep_prob: 1.0})
                print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
                    loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))

                """
                test_output.append(acc_val)
                # test_output_mlp.append(acc_val_mlp)
                # test_output_shortcut_mlp.append(acc_val_shortcut_mlp)
            nn.writer.add_summary(summary, loop_count)
            # mlp.writer.add_summary(summary_mlp, loop_count)
            # shortcut_mlp.writer.add_summary(summary_shortcut_mlp, loop_count)
            """
            if train_loss < 0.01:
                # test_data
                print("------test_data------")
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0})
                print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))

                summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
                    [mlp.summary, mlp.loss, mlp.accuracy],
                    feed_dict={mlp.x: test_data, mlp.t: test_labels, mlp.keep_prob: 1.0})
                print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

                summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
                    [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
                    feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
                print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
                    loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))
                print("------test__end------")
                break
            """
            """for gate_count in range(j):
                gate_value[gate_count] *= 0.999
            """
            i += 1
            loop_count += 1
        # test_data
        print("------test_data------")
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0,
                       nn.gate_9: gate_value[0],
                       nn.gate_8: gate_value[1],
                       nn.gate_7: gate_value[2],
                       nn.gate_6: gate_value[3]})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))

        """
        summary_mlp, loss_val_mlp, acc_val_mlp = mlp.sess.run(
            [mlp.summary, mlp.loss, mlp.accuracy],
            feed_dict={mlp.x: test_data, mlp.t: test_labels, mlp.keep_prob: 1.0})
        print('Step: %d, Loss(ml): %f, Accuracy: %f' % (loop_count, loss_val_mlp, acc_val_mlp))

        summary_shortcut_mlp, loss_val_shortcut_mlp, acc_val_shortcut_mlp = shortcut_mlp.sess.run(
            [shortcut_mlp.summary, shortcut_mlp.loss, shortcut_mlp.accuracy],
            feed_dict={shortcut_mlp.x: test_data, shortcut_mlp.t: test_labels, shortcut_mlp.keep_prob: 1.0})
        print('Step: %d, Loss(sh): %f, Accuracy: %f' % (
            loop_count, loss_val_shortcut_mlp, acc_val_shortcut_mlp))
        """
        print("------test__end------")
        # gate_value[j] = train_accracy
        j += 1
    print("max_test_accuracy")
    print("te {0}:{1}".format(max(test_output),
                              [i * 100 / loop_len for i, x in enumerate(test_output) if x == max(test_output)]))
    """print("ml {0}:{1}".format(max(test_output_mlp),

                              [i * 100 / loop_len for i, x in enumerate(test_output_mlp) if x == max(test_output_mlp)]))
    print("sh {0}:{1}".format(max(test_output_shortcut_mlp),
                              [i * 100 / loop_len for i, x in enumerate(test_output_shortcut_mlp) if
                               x == max(test_output_shortcut_mlp)]))
    """
    print("max_train_accuracy")
    print("tr {0}:{1}".format(max(train_test_output), [i * 100 / loop_len for i, x in enumerate(train_test_output) if
                                                       x == max(train_test_output)]))
    """print("ml {0}:{1}".format(max(train_test_output_mlp),

                              [i * 100 / loop_len for i, x in enumerate(train_test_output_mlp) if
                               x == max(train_test_output_mlp)]))
    print("sh {0}:{1}".format(max(train_test_output_shortcut_mlp),
                              [i * 100 / loop_len for i, x in enumerate(train_test_output_shortcut_mlp) if
                               x >= max(train_test_output_shortcut_mlp) - 0.001]))
    """
    print("train_size:{0}".format(train_size))

    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print(max_accuracy_list)
    print(max_accuracy_layer_list)
    print("dropout:{0}".format(drop_out_rate))
    print("hidden_nodes:{0}".format(nn.hidden_unit))

    max_accuracy_list_with_hyper = list()
    max_accuracy_hyper_parameter = list()
    max_accuracy_with_hyper = 0.0
    split_size = 10.0
    # test_data
    """
    print("------gate_parameter_tuning------")
    for i in range(int(split_size + 1)):
        for j in range(int(split_size + 1)):
            for k in range(int(split_size + 1)):
                for l in range(int(split_size + 1)):
                    summary, loss_val, acc_val = nn.sess.run(
                        [nn.summary, nn.loss[3], nn.accuracy[3]],
                        feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0,
                                   nn.gate_9: i / split_size, nn.gate_8: j / split_size,
                                   nn.gate_7: k / split_size, nn.gate_6: l / split_size})
                    if acc_val > max_accuracy_with_hyper:
                        max_accuracy_with_hyper = acc_val
                        tmp_max = nn.sess.run(nn.accuracy[3],
                                              feed_dict={nn.x: test_data,
                                                         nn.t: test_labels,
                                                         nn.keep_prob: 1.0,
                                                         nn.gate_9: i / split_size,
                                                         nn.gate_8: j / split_size,
                                                         nn.gate_7: k / split_size,
                                                         nn.gate_6: l / split_size})
                        max_accuracy_list_with_hyper.append(tmp_max)
                        max_accuracy_hyper_parameter.append(
                            [i / split_size, j / split_size, k / split_size, l / split_size])
                        print("-----max_in_valid_for_test_data----------------------------------")
                        print('Step: %s, Accuracy: %f'
                              % ([i / split_size, j / split_size, k / split_size, l / split_size], tmp_max))
                        print("----------end----------------------------------------------------")
                    else:
                        print('Step: %s, Loss(va): %f, Accuracy: %f'
                              % ([i / split_size, j / split_size, k / split_size, l / split_size], loss_val, acc_val))
    print(max_accuracy_list_with_hyper)
    print(max_accuracy_hyper_parameter)
    """
    print("train_size:{0}".format(train_size))
    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print("dropout:{0}".format(drop_out_rate))
    print("hidden_nodes:{0}".format(nn.hidden_unit))
    print("not_tuned_model:{0}".format(max_accuracy_list))
    print("not_tuned_model:{0}".format(max_accuracy_layer_list))
    print("valid_size:{0}".format(len(valid_data)))
    print("early_stopping_num:{0}".format(early_stopping_num))

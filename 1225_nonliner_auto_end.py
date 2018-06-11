import tensorflow as tf
import numpy as np
import random
import math
import os
import xlwt
import matplotlib.pyplot as plt
from os import path

# import load_notMNIST_1031

# np.random.seed(20160612)
# tf.set_random_seed(20160612)
np.random.seed(20171109)
tf.set_random_seed(20171109)

image_size = 28 * 28 * 1
output_size = 10
batchsize = 10
hidden_unit_number = 500
input_channel = 1
input_width = int(np.sqrt(image_size))

dataset_flag = 0
bias_ReLU = 0.1#0.01


# ネットワーク構成
class layer:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input_layer'):
            with tf.name_scope('input_size'):
                input_size = image_size * input_channel
            with tf.name_scope('x'):
                x = tf.placeholder(tf.float32, [None, input_size])

        with tf.name_scope('keep_prob'):
            keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('training_flag'):
            training = tf.placeholder(tf.bool)
        with tf.name_scope('training_flag'):
            input_batch_size = tf.placeholder(tf.float32)
        with tf.name_scope('node_size'):
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
                num_units_inner = 500# 100

        with tf.name_scope('layer8'):
            with tf.name_scope('w8'):
                w8 = tf.Variable(
                    tf.random_uniform(shape=[input_size, num_units8],
                                      minval=-np.sqrt(6.0 / (input_size + num_units8)),
                                      maxval=np.sqrt(6.0 / (input_size + num_units8)),
                                      dtype=tf.float32))
            with tf.name_scope('b8'):
                b8 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units8]))
        with tf.name_scope('layer7'):
            with tf.name_scope('w7'):
                w7 = tf.Variable(
                    tf.random_uniform(shape=[num_units8, num_units7],
                                      minval=-np.sqrt(6.0 / (num_units8 + num_units7)),
                                      maxval=np.sqrt(6.0 / (num_units8 + num_units7)),
                                      dtype=tf.float32))
            with tf.name_scope('b7'):
                b7 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units7]))
        with tf.name_scope('layer6'):
            with tf.name_scope('w6'):
                w6 = tf.Variable(
                    tf.random_uniform(shape=[num_units7, num_units6],
                                      minval=-np.sqrt(6.0 / (num_units7 + num_units6)),
                                      maxval=np.sqrt(6.0 / (num_units7 + num_units6)),
                                      dtype=tf.float32))
            with tf.name_scope('b6'):
                b6 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units6]))
        with tf.name_scope('layer5'):
            with tf.name_scope('w5'):
                w5 = tf.Variable(
                    tf.random_uniform(shape=[num_units6, num_units5],
                                      minval=-np.sqrt(6.0 / (num_units6 + num_units5)),
                                      maxval=np.sqrt(6.0 / (num_units6 + num_units5)),
                                      dtype=tf.float32))
            with tf.name_scope('b5'):
                b5 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units5]))
        with tf.name_scope('layer4'):
            with tf.name_scope('w4'):
                w4 = tf.Variable(
                    tf.random_uniform(shape=[num_units5, num_units4],
                                      minval=-np.sqrt(6.0 / (num_units5 + num_units4)),
                                      maxval=np.sqrt(6.0 / (num_units5 + num_units4)),
                                      dtype=tf.float32))
            with tf.name_scope('b4'):
                b4 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units4]))
        with tf.name_scope('layer3'):
            with tf.name_scope('w3'):
                w3 = tf.Variable(
                    tf.random_uniform(shape=[num_units4, num_units3],
                                      minval=-np.sqrt(6.0 / (num_units4 + num_units3)),
                                      maxval=np.sqrt(6.0 / (num_units4 + num_units3)),
                                      dtype=tf.float32))
            with tf.name_scope('b3'):
                b3 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units3]))
        with tf.name_scope('layer2'):
            with tf.name_scope('w2'):
                w2 = tf.Variable(
                    tf.random_uniform(shape=[num_units3, num_units2],
                                      minval=-np.sqrt(6.0 / (num_units3 + num_units2)),
                                      maxval=np.sqrt(6.0 / (num_units3 + num_units2)),
                                      dtype=tf.float32))
            with tf.name_scope('b2'):
                b2 = tf.Variable(tf.constant(bias_ReLU, shape=[num_units2]))

        with tf.name_scope('layer0_shortcut'):
            with tf.name_scope('w0_9'):
                w0_9 = tf.Variable(
                    tf.random_uniform(shape=[input_size, output_size],
                                      minval=-np.sqrt(3.0 / (input_size + output_size)),
                                      maxval=np.sqrt(3.0 / (input_size + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w0_8'):
                w0_8 = tf.Variable(
                    tf.random_uniform(shape=[num_units8, output_size],
                                      minval=-np.sqrt(3.0 / (num_units8 + output_size)),
                                      maxval=np.sqrt(3.0 / (num_units8 + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w0_7'):
                w0_7 = tf.Variable(
                    tf.random_uniform(shape=[num_units7, output_size],
                                      minval=-np.sqrt(3.0 / (num_units7 + output_size)),
                                      maxval=np.sqrt(3.0 / (num_units7 + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w0_6'):
                w0_6 = tf.Variable(tf.random_uniform(shape=[num_units6, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units6 + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units6 + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_5'):
                w0_5 = tf.Variable(tf.random_uniform(shape=[num_units5, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units5 + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units5 + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_4'):
                w0_4 = tf.Variable(tf.random_uniform(shape=[num_units4, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units4 + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units4 + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_3'):
                w0_3 = tf.Variable(tf.random_uniform(shape=[num_units3, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units3 + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units3 + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_2'):
                w0_2 = tf.Variable(tf.random_uniform(shape=[num_units2, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units2 + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units2 + output_size)),
                                                     dtype=tf.float32))

            with tf.name_scope('b0'):
                b0 = tf.Variable(tf.zeros([output_size]))
                b0_8 = tf.Variable(tf.zeros([output_size]))
                b0_7 = tf.Variable(tf.zeros([output_size]))
                b0_6 = tf.Variable(tf.zeros([output_size]))
                b0_5 = tf.Variable(tf.zeros([output_size]))
                b0_4 = tf.Variable(tf.zeros([output_size]))
                b0_3 = tf.Variable(tf.zeros([output_size]))
                b0_2 = tf.Variable(tf.zeros([output_size]))

                # gate_9 = tf.Variable(tf.constant(1.0, shape=[input_size, output_size]))
                # gate_8 = tf.Variable(tf.constant(1.0, shape=[num_units8, output_size]))
                # gate_7 = tf.Variable(tf.constant(1.0, shape=[num_units7, output_size]))
                # ate_6 = tf.Variable(tf.constant(1.0, shape=[num_units6, output_size]))
                gate_9 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_8 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_7 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_6 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_5 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_4 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_3 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_2 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_1 = tf.Variable(tf.constant(1.0, shape=[1]))
        with tf.name_scope('layer_inner_shortcut'):
            w0_inner_8 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_7 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_6 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_5 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_4 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_3 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))
            w0_inner_2 = tf.Variable(
                tf.random_uniform(shape=[num_units_inner, output_size],
                                  minval=-np.sqrt(3.0 / (num_units_inner + output_size)),
                                  maxval=np.sqrt(3.0 / (num_units_inner + output_size)),
                                  dtype=tf.float32))

            w_inner_8 = tf.Variable(
                tf.random_uniform(shape=[num_units8, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units8 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units8 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_7 = tf.Variable(
                tf.random_uniform(shape=[num_units7, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units7 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units7 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_6 = tf.Variable(
                tf.random_uniform(shape=[num_units6, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units6 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units6 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_5 = tf.Variable(
                tf.random_uniform(shape=[num_units5, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units5 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units5 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_4 = tf.Variable(
                tf.random_uniform(shape=[num_units4, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units4 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units4 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_3 = tf.Variable(
                tf.random_uniform(shape=[num_units3, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units3 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units3 + num_units_inner)),
                                  dtype=tf.float32))
            w_inner_2 = tf.Variable(
                tf.random_uniform(shape=[num_units2, num_units_inner],
                                  minval=-np.sqrt(6.0 / (num_units2 + num_units_inner)),
                                  maxval=np.sqrt(6.0 / (num_units2 + num_units_inner)),
                                  dtype=tf.float32))

            with tf.name_scope('b0'):
                b8_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b7_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b6_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b5_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b4_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b3_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))
                b2_inner = tf.Variable(tf.constant(bias_ReLU, shape=[num_units_inner]))

        with tf.name_scope('feed_forword'):
            hidden_8 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x, w8) + b8, training=training))
            hidden_7 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_8, w7) + b7, training=training))
            hidden_6 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_7, w6) + b6, training=training))
            hidden_5 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_6, w5) + b5, training=training))
            hidden_4 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_5, w4) + b4, training=training))
            hidden_3 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_4, w3) + b3, training=training))
            hidden_2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_3, w2) + b2, training=training))

            hidden_inner_8 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_8, w_inner_8) + b8_inner, training=training))
            hidden_inner_7 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_7, w_inner_7) + b7_inner, training=training))
            hidden_inner_6 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_6, w_inner_6) + b6_inner, training=training))
            hidden_inner_5 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_5, w_inner_5) + b5_inner, training=training))
            hidden_inner_4 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_4, w_inner_4) + b4_inner, training=training))
            hidden_inner_3 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_3, w_inner_3) + b3_inner, training=training))
            hidden_inner_2 = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_2, w_inner_2) + b2_inner, training=training))

            p0 = tf.nn.softmax(tf.matmul(hidden_inner_8, w0_inner_8) + b0_8)
            p1 = tf.nn.softmax(tf.matmul(hidden_inner_7, w0_inner_7) + b0_7)
            p2 = tf.nn.softmax(tf.matmul(hidden_inner_6, w0_inner_6) + b0_6)
            p3 = tf.nn.softmax(tf.matmul(hidden_inner_5, w0_inner_5) + b0_5)
            p4 = tf.nn.softmax(tf.matmul(hidden_inner_4, w0_inner_4) + b0_4)
            p5 = tf.nn.softmax(tf.matmul(hidden_inner_3, w0_inner_3) + b0_3)
            p6 = tf.nn.softmax(tf.matmul(hidden_inner_2, w0_inner_2) + b0_2)
            t = tf.placeholder(tf.float32, [None, output_size])

        with tf.name_scope('optimizer'):
            loss0 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0))) / input_batch_size
            loss1 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0))) / input_batch_size
            loss2 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p2, 1e-10, 1.0))) / input_batch_size
            loss3 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p3, 1e-10, 1.0))) / input_batch_size
            loss4 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p4, 1e-10, 1.0))) / input_batch_size
            loss5 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p5, 1e-10, 1.0))) / input_batch_size
            loss6 = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p6, 1e-10, 1.0))) / input_batch_size
            """
            loss0 = -tf.reduce_sum(t - p0) / input_batch_size
            loss1 = -tf.reduce_sum(t - p1) / input_batch_size
            loss2 = -tf.reduce_sum(t - p2) / input_batch_size
            loss3 = -tf.reduce_sum(t - p3) / input_batch_size
            loss4 = -tf.reduce_sum(t - p4) / input_batch_size
            loss5 = -tf.reduce_sum(t - p5) / input_batch_size
            loss6 = -tf.reduce_sum(t - p6) / input_batch_size
            """

            loss = [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
        with tf.name_scope('train_step0'):
            # learning_rate = 0.001  # 0.0001
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step0 = tf.train.AdamOptimizer().minimize(loss0,
                                                                var_list=[w8, b8, w_inner_8, b8_inner, w0_inner_8,
                                                                          b0_8])
                train_step1 = tf.train.AdamOptimizer().minimize(loss1,
                                                                var_list=[w7, b7, w_inner_7, b7_inner, w0_inner_7,
                                                                          b0_7])
                train_step2 = tf.train.AdamOptimizer().minimize(loss2,
                                                                var_list=[w6, b6, w_inner_6, b6_inner, w0_inner_6,
                                                                          b0_6])
                train_step3 = tf.train.AdamOptimizer().minimize(loss3,
                                                                var_list=[w5, b5, w_inner_5, b5_inner, w0_inner_5,
                                                                          b0_5])
                train_step4 = tf.train.AdamOptimizer().minimize(loss4,
                                                                var_list=[w4, b4, w_inner_4, b4_inner, w0_inner_4,
                                                                          b0_4])
                train_step5 = tf.train.AdamOptimizer().minimize(loss5,
                                                                var_list=[w3, b3, w_inner_3, b3_inner, w0_inner_3,
                                                                          b0_3])
                train_step6 = tf.train.AdamOptimizer().minimize(loss6,
                                                                var_list=[w2, b2, w_inner_2, b2_inner, w0_inner_2,
                                                                          b0_2])

                train_step0_under = tf.train.AdamOptimizer().minimize(loss0,
                                                                      var_list=[w_inner_8, b8_inner, w0_inner_8, b0_8])
                train_step1_under = tf.train.AdamOptimizer().minimize(loss1,
                                                                      var_list=[w_inner_7, b7_inner, w0_inner_7, b0_7])
                train_step2_under = tf.train.AdamOptimizer().minimize(loss2,
                                                                      var_list=[w_inner_6, b6_inner, w0_inner_6, b0_6])
                train_step3_under = tf.train.AdamOptimizer().minimize(loss3,
                                                                      var_list=[w_inner_5, b5_inner, w0_inner_5, b0_5])
                train_step4_under = tf.train.AdamOptimizer().minimize(loss4,
                                                                      var_list=[w_inner_4, b4_inner, w0_inner_4, b0_4])
                train_step5_under = tf.train.AdamOptimizer().minimize(loss5,
                                                                      var_list=[w_inner_3, b3_inner, w0_inner_3, b0_3])
                train_step6_under = tf.train.AdamOptimizer().minimize(loss6,
                                                                      var_list=[w_inner_2, b2_inner, w0_inner_2, b0_2])

                train_step0_upper = tf.train.AdamOptimizer().minimize(loss0, var_list=[w0_inner_8, b0_8])
                train_step1_upper = tf.train.AdamOptimizer().minimize(loss1, var_list=[w0_inner_7, b0_7])
                train_step2_upper = tf.train.AdamOptimizer().minimize(loss2, var_list=[w0_inner_6, b0_6])
                train_step3_upper = tf.train.AdamOptimizer().minimize(loss3, var_list=[w0_inner_5, b0_5])
                train_step4_upper = tf.train.AdamOptimizer().minimize(loss4, var_list=[w0_inner_4, b0_4])
                train_step5_upper = tf.train.AdamOptimizer().minimize(loss5, var_list=[w0_inner_3, b0_3])
                train_step6_upper = tf.train.AdamOptimizer().minimize(loss6, var_list=[w0_inner_2, b0_2])
                train_step = [train_step0, train_step1, train_step2, train_step3, train_step4,
                              train_step5, train_step6]
                train_step_under = [train_step0_under, train_step1_under, train_step2_under, train_step3_under,
                                    train_step4_under, train_step5_under, train_step6_under]
                train_step_upper = [train_step0_upper, train_step1_upper, train_step2_upper, train_step3_upper,
                                    train_step4_upper, train_step5_upper, train_step6_upper]

        with tf.name_scope('correct_prediction'):
            correct_prediction0 = tf.equal(tf.argmax(p0, 1), tf.argmax(t, 1))
            correct_prediction1 = tf.equal(tf.argmax(p1, 1), tf.argmax(t, 1))
            correct_prediction2 = tf.equal(tf.argmax(p2, 1), tf.argmax(t, 1))
            correct_prediction3 = tf.equal(tf.argmax(p3, 1), tf.argmax(t, 1))
            correct_prediction4 = tf.equal(tf.argmax(p4, 1), tf.argmax(t, 1))
            correct_prediction5 = tf.equal(tf.argmax(p5, 1), tf.argmax(t, 1))
            correct_prediction6 = tf.equal(tf.argmax(p6, 1), tf.argmax(t, 1))
        with tf.name_scope('accuracy'):
            accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
            accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
            accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
            accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
            accuracy6 = tf.reduce_mean(tf.cast(correct_prediction6, tf.float32))
            accuracy = [accuracy0, accuracy1, accuracy2, accuracy3, accuracy4,
                        accuracy5, accuracy6]

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

        tf.summary.scalar("gate_6", tf.reduce_sum(gate_6))
        tf.summary.scalar("gate_7", tf.reduce_sum(gate_7))
        tf.summary.scalar("gate_8", tf.reduce_sum(gate_8))
        tf.summary.scalar("gate_9", tf.reduce_sum(gate_9))
        self.x, self.t = x, t
        self.loss = loss
        self.accuracy = accuracy
        self.total_accuracy = total_accuracy
        self.total_loss = total_loss
        self.train_total_loss = train_total_loss
        self.train_step = train_step
        self.train_step_under = train_step_under
        self.train_step_upper = train_step_upper
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

        self.gate_9 = gate_9
        self.gate_8 = gate_8
        self.gate_7 = gate_7
        self.gate_6 = gate_6
        self.hidden_unit = hidden_unit
        self.input_size = input_size
        self.num_units0 = num_units0
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.num_units3 = num_units3
        self.num_units4 = num_units4
        self.num_units5 = num_units5
        self.num_units6 = num_units6
        self.num_units7 = num_units7
        self.num_units8 = num_units8
        self.output_size = output_size
        self.p0 = p0
        self.training = training
        self.input_batch_size = input_batch_size

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./log/" + path.splitext(path.basename(__file__))[0], sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = tf.train.Saver(max_to_keep=None)


def initialize_MNIST():
    global dataset_flag
    dataset_flag = 0
    global batchsize
    batchsize = 32
    global image_size
    image_size = 28 * 28  # 32 * 32
    global input_channel
    input_channel = 1  # 3
    global hidden_unit_number
    hidden_unit_number = 500
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4


def initialize_fashion_MNIST():
    global dataset_flag
    dataset_flag = 1
    global batchsize
    batchsize = 32
    global image_size
    image_size = 28 * 28  # 32 * 32
    global input_channel
    input_channel = 1  # 3
    global hidden_unit_number
    hidden_unit_number = 500
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4


def initialize_cifar():
    global dataset_flag
    dataset_flag = 2
    global batchsize
    batchsize = 32
    global input_channel
    input_channel = 3  # 1
    global image_size
    image_size = 32 * 32  # 28*28
    global hidden_unit_number
    hidden_unit_number = 500
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4


def initialize_cifar100():
    global dataset_flag
    dataset_flag = 3
    global batchsize
    batchsize = 128# 32
    global input_channel
    input_channel = 3  # 1
    global image_size
    image_size = 32 * 32  # 28*28
    global hidden_unit_number
    hidden_unit_number = 500
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4
    global output_size
    output_size = 100


if __name__ == '__main__':
    initialize_cifar100()
    train_size = 46000  # 10000 is good
    layer_num = 6
    start = layer_num
    start = 0
    early_stopping_num = 10
    drop_out_rate = 1.0

    session_number = 0
    session_name = "./session_log/saver" + path.splitext(path.basename(__file__))[0] + str(session_number)
    session_name_layer = "./session_log/saver" + path.splitext(path.basename(__file__))[0] + str(session_number)
    print(session_name)
    training = True
    testing = False

    nn = layer()
    batch_xs = np.mat([[0.0 for n in range(image_size * input_channel)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(output_size)] for k in range(batchsize)])

    dataset_name = ""
    if dataset_flag == 0:
        data = np.load("MNIST_train_data.npy")[0:train_size]
        labels = np.load("MNIST_train_labels.npy")[0:train_size]
        test_data = np.load("MNIST_test_data.npy")
        test_labels = np.load("MNIST_test_labels.npy")
        valid_data = np.load("MNIST_valid_data.npy")[0:int(train_size / 10)]
        valid_labels = np.load("MNIST_valid_labels.npy")[0:int(train_size / 10)]
        dataset_name = "MNIST"
    if dataset_flag == 1:
        data = np.load("fashion_train_data.npy")[0:train_size]
        labels = np.load("fashion_train_labels.npy")[0:train_size]
        test_data = np.load("fashion_test_data.npy")
        test_labels = np.load("fashion_test_labels.npy")
        valid_data = np.load("fashion_valid_data.npy")[0:int(train_size / 10)]
        valid_labels = np.load("fashion_valid_labels.npy")[0:int(train_size / 10)]
        dataset_name = "fashion"
    if dataset_flag == 2:
        data = np.load("cifar-10_train_data_normalized.npy")[0:train_size]
        labels = np.load("cifar-10_train_labels.npy")[0:train_size]
        test_data = np.load("cifar-10_test_data_normalized.npy")
        test_labels = np.load("cifar-10_test_labels.npy")
        valid_data = np.load("cifar-10_train_data_normalized.npy")[46000:46000 + int(train_size / 10)]
        valid_labels = np.load("cifar-10_train_labels.npy")[46000:46000 + int(train_size / 10)]
        dataset_name = "CIFAR-10"
    if dataset_flag == 3:
        data = np.load("cifar-100_train_data.npy")[0:train_size]
        labels = np.load("cifar-100_train_label.npy")[0:train_size]
        test_data = np.load("cifar-100_test_data.npy")[0:10000]
        test_labels = np.load("cifar-100_test_label.npy")[0:10000]
        valid_data = np.load("cifar-100_train_data.npy")[46000:46000 + int(train_size / 10)]
        valid_labels = np.load("cifar-100_train_label.npy")[46000:46000 + int(train_size / 10)]
        dataset_name = "CIFAR-100"
    print(data.shape)

    train_test_size = train_size
    if train_size > 100:
        train_test_size = 100
    print("train_size:{0}".format(train_size))
    print("train_test_size:{0}".format(train_test_size))
    print("batch_size:{0}".format(batchsize))
    train_list_num = list(range(train_test_size))
    # train_list_num = random.sample(list(range(len(data))), train_size)

    train_test_data = np.mat([[0.0 for n in range(image_size * input_channel)] for k in range(len(train_list_num))])
    train_test_labels = np.mat([[0.0 for n in range(output_size)] for k in range(len(train_list_num))])
    for i in range(len(train_test_data)):
        tmp = train_list_num[i]
        train_test_data[i] = data[tmp].reshape(1, image_size * input_channel)
        train_test_labels[i] = labels[tmp].reshape(1, output_size)

    test_output = list()
    train_test_output = list()
    max_accuracy = 0.0
    max_accuracy_list = list()
    max_accuracy_layer_list = list()

    final_acc_valid = 0
    final_loss_valid = 10000.
    final_accuracy_test = 0
    final_loss_layer = 0
    graph_x = range(1000)
    graph_y = [[None for i in range(1000)] for j in range(layer_num + 2)]
    graph_train_y = [[None for i in range(1000)] for j in range(layer_num + 2)]

    loop_len = 100000  # 400000
    print("loop_len:{0}".format(loop_len))
    loop_count = 0
    acc_val = 0.
    gate_value = [1., 1., 1., 1.]
    print("dropout:{0}".format(drop_out_rate))
    train_accracy = 0.0
    best_loss = 10000.
    stopping_step = 0
    layer_loss = 10000.
    loop_num = 0

    for j in range(start, layer_num + 1):
        # test_data
        print("------test_with_no_train------")
        print(len(valid_labels))
        summary_tmp, loss_val_tmp, acc_val_tmp = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(valid_labels)})
        print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_tmp, acc_val_tmp))
        summary_tmp, loss_val_tmp, acc_val_tmp = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(test_labels)})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val_tmp, acc_val_tmp))
        print("-------------end---------------")

        print("step:{0} start".format(j))
        stopping_step = 0
        """
        if j == 1:
            tmp_array = nn.sess.run(nn.w0_9)
            nn.sess.run(nn.w0_9,
                        feed_dict={nn.w0_9: tmp_array * np.sqrt(nn.input_size+nn.output_size
                                                                / (nn.input_size + nn.hidden_unit+nn.output_size))})
        if j == 2:
            tmp_array = nn.sess.run(nn.w0_9)
            nn.sess.run(nn.w0_9, feed_dict={nn.w0_9: tmp_array * np.sqrt(
                np.sqrt((nn.hidden_unit + nn.output_size)
                        / (nn.hidden_unit * 2 + nn.output_size)))})
            tmp_array = nn.sess.run(nn.w0_8)
            nn.sess.run(nn.w0_8, feed_dict={nn.w0_8: tmp_array * np.sqrt(
                np.sqrt((nn.input_size + nn.hidden_unit + nn.output_size)
                        / (nn.hidden_unit * 2 + nn.output_size)))})
        if j == 3:
            tmp_array = nn.sess.run(nn.w0_9)
            nn.sess.run(nn.w0_9, feed_dict={nn.w0_9: tmp_array * np.sqrt(
                np.sqrt((nn.input_size + nn.hidden_unit*2 + nn.output_size)
                        / (nn.hidden_unit * 3 + nn.output_size)))})
            tmp_array = nn.sess.run(nn.w0_8)
            nn.sess.run(nn.w0_8, feed_dict={nn.w0_8: tmp_array * np.sqrt(
                np.sqrt((nn.input_size + nn.hidden_unit*2 + nn.output_size)
                        / (nn.hidden_unit * 3 + nn.output_size)))})
            tmp_array = nn.sess.run(nn.w0_7)
            nn.sess.run(nn.w0_7, feed_dict={nn.w0_7: tmp_array * np.sqrt(
                np.sqrt((nn.input_size + nn.hidden_unit*2 + nn.output_size)
                        / (nn.hidden_unit * 3 + nn.output_size)))})
        """
        best_loss = 10000.

        for i in range(loop_len):
            each_epoch = list(range(train_size))
            random.shuffle(each_epoch)
            while len(each_epoch) >= batchsize:
                # print(len(each_epoch))
                for n in range(batchsize):
                    tmp = each_epoch[0]
                    each_epoch.remove(tmp)
                    batch_xs[n] = data[tmp].reshape(1, image_size * input_channel)
                    batch_ts[n] = labels[tmp].reshape(1, output_size)
                nn.sess.run(nn.train_step[j], feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                                         nn.training: training, nn.input_batch_size: batchsize})

            tmp_loss = nn.sess.run(nn.loss[j], feed_dict={nn.x: valid_data, nn.t: valid_labels,
                                                          nn.keep_prob: 1.0, nn.training: testing,
                                                          nn.input_batch_size: len(valid_labels)})
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
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                               nn.input_batch_size: train_test_size})
                print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val, nn.training: testing})
                graph_train_y[j][loop_count] = acc_val
                train_loss = loss_val
                train_accracy = acc_val
                nn.sess.run(nn.assign_op_train_total_loss,
                            feed_dict={nn.input_placeholder_train_total_loss: loss_val, nn.training: testing})
                train_test_output.append(acc_val)
                # valid_data
                loss_val, acc_val = nn.sess.run(
                    [nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                               nn.input_batch_size: len(valid_labels)})
                print('Step(%d): %d, Loss(va): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))

                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                               nn.input_batch_size: len(test_labels)})
                print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                test_output.append(acc_val)
                graph_y[j][loop_count] = acc_val
            nn.writer.add_summary(summary, loop_count)
            i += 1
            loop_count += 1

        # test_data
        print("------test_data------")
        summary, loss_val_train, acc_val_train = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(train_test_labels)})
        print('Step: %d, Loss(tr): %f, Accuracy: %f' % (loop_count, loss_val_train, acc_val_train))
        summary, loss_val_valid, acc_val_valid = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(valid_labels)})
        print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_valid, acc_val_valid))
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(test_labels)})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
        if acc_val_valid > final_acc_valid:
            final_acc_valid = acc_val_valid
            final_accuracy_test = acc_val
            print("------test__end------")
        if loss_val_valid*1.001 < final_loss_valid:
            final_loss_valid = loss_val_valid
            loop_num = loop_count
            final_loss_layer = j
            nn.saver.save(nn.sess, session_name_layer)
        else: # when loss doesn't improved
            nn.saver.restore(nn.sess, session_name_layer)
            nn.saver.save(nn.sess, session_name)
            for reset_num in range(early_stopping_num):
                graph_y[j][loop_count + 1 + reset_num] = None
                graph_train_y[j][loop_count + 1 + reset_num] = None
            if j < layer_num:
                graph_y[j + 1][loop_count] = acc_val
                graph_train_y[j + 1][loop_count] = acc_val_train
            break
        for reset_num in range(early_stopping_num):
            graph_y[j][loop_count + 1 + reset_num] = None
            graph_train_y[j][loop_count + 1 + reset_num] = None
        if j < layer_num:
            graph_y[j + 1][loop_count] = acc_val
            graph_train_y[j + 1][loop_count] = acc_val_train
        j += 1
        loop_count += 1

    print("")
    print("automatically_finish_triggered")
    print("")
    # train_under_and_upper_weight
    print("under_weight_learn_start")
    best_loss = 10000.
    loop_count = loop_num + 1
    for i in range(loop_len):
        each_epoch = list(range(train_size))
        random.shuffle(each_epoch)
        while len(each_epoch) >= batchsize:
            for n in range(batchsize):
                tmp = each_epoch[0]
                each_epoch.remove(tmp)
                batch_xs[n] = data[tmp].reshape(1, image_size * input_channel)
                batch_ts[n] = labels[tmp].reshape(1, output_size)
            nn.sess.run(nn.train_step_under[final_loss_layer],
                        feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                   nn.training: training, nn.input_batch_size: batchsize})
        tmp_loss = nn.sess.run(nn.loss[final_loss_layer],
                               feed_dict={nn.x: valid_data, nn.t: valid_labels,
                                          nn.keep_prob: 1.0, nn.training: testing,
                                          nn.input_batch_size: len(valid_labels)})
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
                [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: train_test_size})
            print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))
            nn.sess.run(nn.assign_op_train_total_accuracy,
                        feed_dict={nn.input_placeholder_train_total_accuracy: acc_val, nn.training: testing})
            graph_train_y[final_loss_layer][loop_count] = acc_val
            train_loss = loss_val
            train_accracy = acc_val
            nn.sess.run(nn.assign_op_train_total_loss,
                        feed_dict={nn.input_placeholder_train_total_loss: loss_val, nn.training: testing})
            train_test_output.append(acc_val)
            # valid_data
            loss_val, acc_val = nn.sess.run(
                [nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: len(valid_labels)})
            print('Step(%d): %d, Loss(va): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))

            # test_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: len(test_labels)})
            print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))
            test_output.append(acc_val)
            graph_y[final_loss_layer][loop_count] = acc_val
        nn.writer.add_summary(summary, loop_count)
        i += 1
        loop_count += 1
    # test_data
    print("------test_data------")
    summary, loss_val_train, acc_val_train = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(train_test_labels)})
    print('Step: %d, Loss(tr): %f, Accuracy: %f' % (loop_count, loss_val_train, acc_val_train))
    summary, loss_val_valid, acc_val_valid = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(valid_labels)})
    print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_valid, acc_val_valid))
    summary, loss_val, acc_val = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(test_labels)})
    print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))

    loop_count += 1
    print("upper_weight_learn_start")
    for i in range(loop_len):
        each_epoch = list(range(train_size))
        random.shuffle(each_epoch)
        while len(each_epoch) >= batchsize:
            for n in range(batchsize):
                tmp = each_epoch[0]
                each_epoch.remove(tmp)
                batch_xs[n] = data[tmp].reshape(1, image_size * input_channel)
                batch_ts[n] = labels[tmp].reshape(1, output_size)
            nn.sess.run(nn.train_step_upper[final_loss_layer],
                        feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                   nn.training: training, nn.input_batch_size: batchsize})
        tmp_loss = nn.sess.run(nn.loss[final_loss_layer],
                               feed_dict={nn.x: valid_data, nn.t: valid_labels,
                                          nn.keep_prob: 1.0, nn.training: testing,
                                          nn.input_batch_size: len(valid_labels)})
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
                [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: train_test_size})
            print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))
            nn.sess.run(nn.assign_op_train_total_accuracy,
                        feed_dict={nn.input_placeholder_train_total_accuracy: acc_val, nn.training: testing})
            graph_train_y[final_loss_layer+1][loop_count] = acc_val
            train_loss = loss_val
            train_accracy = acc_val
            nn.sess.run(nn.assign_op_train_total_loss,
                        feed_dict={nn.input_placeholder_train_total_loss: loss_val, nn.training: testing})
            train_test_output.append(acc_val)
            # valid_data
            loss_val, acc_val = nn.sess.run(
                [nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: len(valid_labels)})
            print('Step(%d): %d, Loss(va): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))

            # test_data
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
                feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                           nn.input_batch_size: len(test_labels)})
            print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (final_loss_layer, loop_count, loss_val, acc_val))
            test_output.append(acc_val)
            graph_y[final_loss_layer+1][loop_count] = acc_val
        nn.writer.add_summary(summary, loop_count)
        i += 1
        loop_count += 1

    # test_data
    print("------test_data------")
    summary, loss_val_train, acc_val_train = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(train_test_labels)})
    print('Step: %d, Loss(tr): %f, Accuracy: %f' % (loop_count, loss_val_train, acc_val_train))
    summary, loss_val_valid, acc_val_valid = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(valid_labels)})
    print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_valid, acc_val_valid))
    summary, loss_val, acc_val = nn.sess.run(
        [nn.summary, nn.loss[final_loss_layer], nn.accuracy[final_loss_layer]],
        feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                   nn.input_batch_size: len(test_labels)})
    print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
    loop_count += 1

    print("max_test_accuracy")
    print("te {0}:{1}".format(max(test_output),
                              [i * 100 for i, x in enumerate(test_output) if x == max(test_output)]))
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
    print("train_size:{0}".format(train_size))
    print("batch_size:{0}".format(batchsize))
    print("loop_count:{0}".format(loop_count))
    print("dropout:{0}".format(drop_out_rate))
    print("hidden_nodes:{0}".format(nn.hidden_unit))
    print("not_tuned_model:{0}".format(max_accuracy_list))
    print("not_tuned_model:{0}".format(max_accuracy_layer_list))
    print("valid_size:{0}".format(len(valid_data)))
    print("early_stopping_num:{0}".format(early_stopping_num))
    print("final_accuracy:{0}".format(final_accuracy_test))
    print("final_accuracy's_hidden_layer_num:{0}".format(final_loss_layer))
    print(loop_count)
    for i in range(early_stopping_num):
        graph_y[-1][loop_count + i] = None
        graph_train_y[-1][loop_count + i] = None

    ax = plt.subplot()
    for i in range(layer_num + 1):
        ax.plot(graph_x, graph_y[-i], linewidth=1.0, marker=".", markersize=10)
    for i in range(layer_num + 1):
        ax.plot(graph_x, graph_train_y[-i], linewidth=1.0, marker=".", markersize=10)
    ax.grid(which="both")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.title(str(train_size) + dataset_name + ":" + path.splitext(path.basename(__file__))[0])
    ax.set_xlim([0, loop_count - 1])

    np.save("MLP_graph_y", graph_y)
    np.save("graph_x", graph_x)

    plt.show()

    data_nonliner = np.load("nonliner_graph_y.npy")

    ax = plt.subplot()
    for i in range(layer_num + 2):
        ax.plot(graph_x, graph_y[-i], linewidth=1.0, marker=".", markersize=10)
    for i in range(layer_num + 2):
        ax.plot(graph_x, data_nonliner[-i], linewidth=1.0, marker=",", markersize=10)
    ax.grid(which="both")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.title(str(train_size) + dataset_name + ":" + path.splitext(path.basename(__file__))[0])
    ax.set_xlim([0, loop_count - 1])
    plt.show()

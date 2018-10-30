import tensorflow as tf
import numpy as np
import random
import math
import os
import xlwt
import matplotlib.pyplot as plt
import sys

from os import path
from matplotlib.font_manager import FontProperties
font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

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
                num_units8 = 2000
                num_units7 = 1000
                num_units6 = 500
                num_units5 = 30
                """
                num_units8 = int(hidden_unit)
                num_units7 = int(hidden_unit)
                num_units6 = int(hidden_unit)
                num_units5 = int(hidden_unit)
                """
                num_units4 = int(hidden_unit)
                num_units3 = int(hidden_unit)
                num_units2 = int(hidden_unit)
                num_units1 = int(hidden_unit)
                num_units0 = output_size
                num_units_insert = 1000

        with tf.name_scope('layer8'):
            with tf.name_scope('w8'):
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
                # b1 = tf.Variable(tf.constant(0., shape=[num_units1]))
                b1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))
        with tf.name_scope('layer0'):
            with tf.name_scope('w0'):
                w0 = tf.Variable(
                    tf.random_uniform(shape=[num_units1, num_units0],
                                      minval=-np.sqrt(3.0 / (num_units1 + num_units0)),
                                      maxval=np.sqrt(3.0 / (num_units1 + num_units0)),
                                      dtype=tf.float32))

        with tf.name_scope('AE_layer'):
            w8_AE = tf.Variable(
                tf.random_uniform(shape=[num_units8, input_size],
                                  minval=-np.sqrt(6.0 / (input_size + num_units8)),
                                  maxval=np.sqrt(6.0 / (input_size + num_units8)),
                                  dtype=tf.float32))
            b8_AE = tf.Variable(tf.constant(0.1, shape=[input_size]))
        with tf.name_scope('layer7'):
            w7_AE = tf.Variable(
                tf.random_uniform(shape=[num_units7, num_units8],
                                  minval=-np.sqrt(6.0 / (num_units8 + num_units7)),
                                  maxval=np.sqrt(6.0 / (num_units8 + num_units7)),
                                  dtype=tf.float32))
            b7_AE = tf.Variable(tf.constant(0.1, shape=[num_units8]))
        with tf.name_scope('layer6'):
            w6_AE = tf.Variable(
                tf.random_uniform(shape=[num_units6, num_units7],
                                  minval=-np.sqrt(6.0 / (num_units7 + num_units6)),
                                  maxval=np.sqrt(6.0 / (num_units7 + num_units6)),
                                  dtype=tf.float32))
            b6_AE = tf.Variable(tf.constant(0.1, shape=[num_units7]))
        with tf.name_scope('layer5'):
            w5_AE = tf.Variable(
                tf.random_uniform(shape=[num_units5, num_units6],
                                  minval=-np.sqrt(6.0 / (num_units6 + num_units5)),
                                  maxval=np.sqrt(6.0 / (num_units6 + num_units5)),
                                  dtype=tf.float32))
            b5_AE = tf.Variable(tf.constant(0.1, shape=[num_units6]))
        with tf.name_scope('layer4'):
            w4_AE = tf.Variable(
                tf.random_uniform(shape=[num_units4, num_units5],
                                  minval=-np.sqrt(6.0 / (num_units5 + num_units4)),
                                  maxval=np.sqrt(6.0 / (num_units5 + num_units4)),
                                  dtype=tf.float32))
            b4_AE = tf.Variable(tf.constant(0.1, shape=[num_units5]))
        with tf.name_scope('layer3'):
            w3_AE = tf.Variable(
                tf.random_uniform(shape=[num_units3, num_units4],
                                  minval=-np.sqrt(6.0 / (num_units4 + num_units3)),
                                  maxval=np.sqrt(6.0 / (num_units4 + num_units3)),
                                  dtype=tf.float32))
            b3_AE = tf.Variable(tf.constant(0.1, shape=[num_units4]))
        with tf.name_scope('layer2'):
            w2_AE = tf.Variable(
                tf.random_uniform(shape=[num_units2, num_units3],
                                  minval=-np.sqrt(6.0 / (num_units3 + num_units2)),
                                  maxval=np.sqrt(6.0 / (num_units3 + num_units2)),
                                  dtype=tf.float32))
            b2_AE = tf.Variable(tf.constant(0.1, shape=[num_units3]))
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
        with tf.name_scope('layer0_shortcut'):
            with tf.name_scope('w0_8'):
                w0_8 = tf.Variable(
                    tf.random_uniform(shape=[num_units_insert, output_size],
                                      minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                      maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w0_7'):
                w0_7 = tf.Variable(
                    tf.random_uniform(shape=[num_units_insert, output_size],
                                      minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                      maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                      dtype=tf.float32))
            with tf.name_scope('w0_6'):
                w0_6 = tf.Variable(tf.random_uniform(shape=[num_units_insert, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_5'):
                w0_5 = tf.Variable(tf.random_uniform(shape=[num_units_insert, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_4'):
                w0_4 = tf.Variable(tf.random_uniform(shape=[num_units_insert, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_3'):
                w0_3 = tf.Variable(tf.random_uniform(shape=[num_units_insert, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_2'):
                w0_2 = tf.Variable(tf.random_uniform(shape=[num_units_insert, output_size],
                                                     minval=-np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     maxval=np.sqrt(3.0 / (num_units_insert + output_size)),
                                                     dtype=tf.float32))
            with tf.name_scope('w0_8'):
                w_insert_8 = tf.Variable(tf.random_uniform(shape=[num_units8, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units8 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units8 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_7'):
                w_insert_7 = tf.Variable(tf.random_uniform(shape=[num_units7, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units7 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units7 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_6'):
                w_insert_6 = tf.Variable(tf.random_uniform(shape=[num_units6, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units6 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units6 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_5'):
                w_insert_5 = tf.Variable(tf.random_uniform(shape=[num_units5, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units5 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units5 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_4'):
                w_insert_4 = tf.Variable(tf.random_uniform(shape=[num_units4, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units4 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units4 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_3'):
                w_insert_3 = tf.Variable(tf.random_uniform(shape=[num_units3, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units3 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units3 + num_units_insert)),
                                                           dtype=tf.float32))
            with tf.name_scope('w0_2'):
                w_insert_2 = tf.Variable(tf.random_uniform(shape=[num_units2, num_units_insert],
                                                           minval=-np.sqrt(3.0 / (num_units2 + num_units_insert)),
                                                           maxval=np.sqrt(3.0 / (num_units2 + num_units_insert)),
                                                           dtype=tf.float32))
            b_insert_8 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_7 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_6 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_5 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_4 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_3 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))
            b_insert_2 = tf.Variable(tf.constant(0.1, shape=[num_units_insert]))

        with tf.name_scope('feed_forword'):
            hidden_8 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x, w8) + b8, training=training))
            hidden_7 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_8, w7) + b7, training=training))
            hidden_6 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_7, w6) + b6, training=training))
            hidden_5 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_6, w5) + b5, training=training))
            hidden_4 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_5, w4) + b4, training=training))
            hidden_3 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_4, w3) + b3, training=training))
            hidden_2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(hidden_3, w2) + b2, training=training))

            """
            p0 = tf.nn.relu(tf.matmul(hidden_8, w8_AE) + b8_AE)
            p1 = tf.nn.relu(tf.matmul(hidden_7, w7_AE) + b7_AE)
            p2 = tf.nn.relu(tf.matmul(hidden_6, w6_AE) + b6_AE)
            p3 = tf.nn.relu(tf.matmul(hidden_5, w5_AE) + b5_AE)
            p4 = tf.nn.relu(tf.matmul(hidden_4, w4_AE) + b4_AE)
            p5 = tf.nn.relu(tf.matmul(hidden_3, w3_AE) + b3_AE)
            p6 = tf.nn.relu(tf.matmul(hidden_2, w2_AE) + b2_AE)
            """
            hidden_8_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_8, w_insert_8) + b_insert_8, training=training))
            hidden_7_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_7, w_insert_7) + b_insert_7, training=training))
            hidden_6_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_6, w_insert_6) + b_insert_6, training=training))
            hidden_5_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_5, w_insert_5) + b_insert_5, training=training))
            hidden_4_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_4, w_insert_4) + b_insert_4, training=training))
            hidden_3_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_3, w_insert_3) + b_insert_3, training=training))
            hidden_2_test = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_2, w_insert_2) + b_insert_2, training=training))

            p0_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_8, w8_AE) + b8_AE, training=training))
            p1_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_7, w7_AE) + b7_AE, training=training))
            p2_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_6, w6_AE) + b6_AE, training=training))
            p3_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_5, w5_AE) + b5_AE, training=training))
            p4_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_4, w4_AE) + b4_AE, training=training))
            p5_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_3, w3_AE) + b3_AE, training=training))
            p6_train = tf.nn.relu(
                tf.layers.batch_normalization(tf.matmul(hidden_2, w2_AE) + b2_AE, training=training))
            p0 = tf.nn.softmax(tf.matmul(hidden_8_test, w0_8) + b0_8)
            p1 = tf.nn.softmax(tf.matmul(hidden_7_test, w0_7) + b0_7)
            p2 = tf.nn.softmax(tf.matmul(hidden_6_test, w0_6) + b0_6)
            p3 = tf.nn.softmax(tf.matmul(hidden_5_test, w0_5) + b0_5)
            p4 = tf.nn.softmax(tf.matmul(hidden_4_test, w0_4) + b0_4)
            p5 = tf.nn.softmax(tf.matmul(hidden_3_test, w0_3) + b0_3)
            p6 = tf.nn.softmax(tf.matmul(hidden_2_test, w0_2) + b0_2)

            t_8 = x
            t_7 = hidden_8
            t_6 = hidden_7
            t_5 = hidden_6
            t_4 = hidden_5
            t_3 = hidden_4
            t_2 = hidden_3

            t = tf.placeholder(tf.float32, [None, output_size])
        with tf.name_scope('optimizer'):
            with tf.name_scope('loss0'):
                loss0 = tf.reduce_sum(tf.square(t_8 - tf.clip_by_value(p0_train, 1e-10, 1.0))) / input_batch_size
                loss1 = tf.reduce_sum(tf.square(t_7 - tf.clip_by_value(p1_train, 1e-10, 1.0))) / input_batch_size
                loss2 = tf.reduce_sum(tf.square(t_6 - tf.clip_by_value(p2_train, 1e-10, 1.0))) / input_batch_size
                loss3 = tf.reduce_sum(tf.square(t_5 - tf.clip_by_value(p3_train, 1e-10, 1.0))) / input_batch_size
                loss4 = tf.reduce_sum(tf.square(t_4 - tf.clip_by_value(p4_train, 1e-10, 1.0))) / input_batch_size
                loss5 = tf.reduce_sum(tf.square(t_3 - tf.clip_by_value(p5_train, 1e-10, 1.0))) / input_batch_size
                loss6 = tf.reduce_sum(tf.square(t_2 - tf.clip_by_value(p6_train, 1e-10, 1.0))) / input_batch_size

                loss0_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p0, 1e-10, 1.0))) / input_batch_size
                loss1_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p1, 1e-10, 1.0))) / input_batch_size
                loss2_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p2, 1e-10, 1.0))) / input_batch_size
                loss3_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p3, 1e-10, 1.0))) / input_batch_size
                loss4_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p4, 1e-10, 1.0))) / input_batch_size
                loss5_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p5, 1e-10, 1.0))) / input_batch_size
                loss6_test = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p6, 1e-10, 1.0))) / input_batch_size

            loss = [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
            loss_test = [loss0_test, loss1_test, loss2_test, loss3_test, loss4_test, loss5_test, loss6_test]
        with tf.name_scope('train_step0'):
            # learning_rate = 0.001  # 0.0001
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step0 = tf.train.AdamOptimizer().minimize(loss0, var_list=[w8, b8, w8_AE, b8_AE])
                train_step1 = tf.train.AdamOptimizer().minimize(loss1, var_list=[w7, b7, w7_AE, b7_AE])
                train_step2 = tf.train.AdamOptimizer().minimize(loss2, var_list=[w6, b6, w6_AE, b6_AE])
                train_step3 = tf.train.AdamOptimizer().minimize(loss3, var_list=[w5, b5, w5_AE, b5_AE])
                train_step4 = tf.train.AdamOptimizer().minimize(loss4, var_list=[w4, b4, w4_AE, b4_AE])
                train_step5 = tf.train.AdamOptimizer().minimize(loss5, var_list=[w3, b3, w3_AE, b3_AE])
                train_step6 = tf.train.AdamOptimizer().minimize(loss6, var_list=[w2, b2, w2_AE, b2_AE])
                train_step = [train_step0, train_step1, train_step2, train_step3, train_step4,
                              train_step5, train_step6]
                train_step0_test = tf.train.AdamOptimizer().minimize(loss0_test)  # , var_list=[w8, b8, w8_AE, b8_AE])
                train_step1_test = tf.train.AdamOptimizer().minimize(loss1_test)  # , var_list=[w7, b7, w7_AE, b7_AE])
                train_step2_test = tf.train.AdamOptimizer().minimize(loss2_test)  # , var_list=[w6, b6, w6_AE, b6_AE])
                train_step3_test = tf.train.AdamOptimizer().minimize(loss3_test)  # , var_list=[w5, b5, w5_AE, b5_AE])
                train_step4_test = tf.train.AdamOptimizer().minimize(loss4_test)  # , var_list=[w4, b4, w4_AE, b4_AE])
                train_step5_test = tf.train.AdamOptimizer().minimize(loss5_test)  # , var_list=[w3, b3, w0_3, b0_3])
                train_step6_test = tf.train.AdamOptimizer().minimize(loss6_test)  # , var_list=[w0, b0, w8, b8])
                train_step_test = [train_step0_test, train_step1_test, train_step2_test, train_step3_test,
                                   train_step4_test,
                                   train_step5_test, train_step6_test]
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
        self.keep_prob = keep_prob
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8

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
        self.loss_test = loss_test
        self.train_step_test = train_step_test

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./log/1105_MNIST_nonliner", sess.graph)
        writer = tf.summary.FileWriter("./log/"+path.splitext(path.basename(__file__))[0], sess.graph)
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
    hidden_unit_number = 1000
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
    args = sys.argv
    if args[1] == "cifar":
        initialize_cifar()
        print("cifar10")
    elif args[1] == "cifar100":
        initialize_cifar100()
        print("cifar100")
    elif args[1] == "mnist":
        initialize_MNIST()
        print("mnist")
    else:
        print("Error")
        print("./1224_... dataset train_size layer_num")
        exit()

    train_size = int(args[2])
    layer_num = int(args[3])

    start = layer_num
    start = 0
    early_stopping_num = 10
    last_early_stopping_num = 30
    drop_out_rate = 1.0
    if len(args) < 5:
        session_number = 0
    else:
        session_number = args[4]
    session_name = "./session_log/saver" + path.splitext(path.basename(__file__))[0] + str(session_number)
    print(session_name)
    training = True
    testing = False

    nn = layer()
    batch_xs = np.mat([[0.0 for n in range(image_size * input_channel)] for k in range(batchsize)])
    # batch_ts = np.mat([[0.0 for n in range(image_size * input_channel)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(output_size)] for k in range(batchsize)])

    if dataset_flag == 0:
        data = np.load("MNIST_train_data.npy")[0:train_size]
        labels = np.load("MNIST_train_labels.npy")[0:train_size]
        test_data = np.load("MNIST_test_data.npy")
        test_labels = np.load("MNIST_test_labels.npy")
        valid_data = np.load("MNIST_valid_data.npy")[0:int(train_size / 10)]
        valid_labels = np.load("MNIST_valid_labels.npy")[0:int(train_size / 10)]
    if dataset_flag == 1:
        data = np.load("fashion_train_data.npy")[0:train_size]
        labels = np.load("fashion_train_labels.npy")[0:train_size]
        test_data = np.load("fashion_test_data.npy")
        test_labels = np.load("fashion_test_labels.npy")
        valid_data = np.load("fashion_valid_data.npy")[0:int(train_size / 10)]
        valid_labels = np.load("fashion_valid_labels.npy")[0:int(train_size / 10)]
    if dataset_flag == 2:
        data = np.load("cifar-10_train_data_normalized.npy")[0:train_size]
        labels = np.load("cifar-10_train_labels.npy")[0:train_size]
        test_data = np.load("cifar-10_test_data_normalized.npy")
        test_labels = np.load("cifar-10_test_labels.npy")
        valid_data = np.load("cifar-10_train_data_normalized.npy")[46000:46000 + int(train_size / 10)]
        valid_labels = np.load("cifar-10_train_labels.npy")[46000:46000 + int(train_size / 10)]
    if dataset_flag == 3:
        data = np.load("cifar-100_train_data.npy")[0:train_size]
        labels = np.load("cifar-100_train_label.npy")[0:train_size]
        test_data = np.load("cifar-100_test_data.npy")[0:10000]
        test_labels = np.load("cifar-100_test_label.npy")[0:10000]
        valid_data = np.load("cifar-100_train_data.npy")[46000:46000 + int(train_size / 10)]
        valid_labels = np.load("cifar-100_train_label.npy")[46000:46000 + int(train_size / 10)]
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
    final_accuracy_test = 0
    final_loss_layer = 0
    graph_x = range(1000)
    graph_y = [None for i in range(1000)]
    graph_train_y = [None for i in range(1000)]

    loop_len = 100000  # 400000
    print("loop_len:{0}".format(loop_len))
    loop_count = 0
    acc_val = 0.
    gate_value = [1., 1., 1., 1.]
    print("dropout:{0}".format(drop_out_rate))
    train_accracy = 0.0
    best_loss = 10000.
    stopping_step = 0

    for j in range(start, layer_num):
        # test_data
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
                                                         nn.training: training,
                                                         nn.input_batch_size: batchsize})

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
                # test_data
                summary, loss_val = nn.sess.run(
                    [nn.summary, nn.loss[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                               nn.input_batch_size: len(test_labels)})
                print('Step(%d): %d, Loss(te): %f' % (j, loop_count, loss_val))
            nn.writer.add_summary(summary, loop_count)
            i += 1
            loop_count += 1
        # test_data
        print("------test_data------")
        summary, loss_val = nn.sess.run(
            [nn.summary, nn.loss[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(test_labels)})
        print('Step: %d, Loss(te): %f' % (loop_count, loss_val))
        j += 1
        loop_count += 1

#initialize for sotsuron
    loop_count = 0
    early_stopping_num = last_early_stopping_num
# initialize for sotsuron

    for j in range(layer_num - 1, layer_num):
        stopping_step = 0
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
                nn.sess.run(nn.train_step_test[j],
                            feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: drop_out_rate,
                                       nn.training: training,
                                       nn.input_batch_size: batchsize})

            tmp_loss = nn.sess.run(nn.loss_test[j], feed_dict={nn.x: valid_data, nn.t: valid_labels,
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
                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss_test[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                               nn.input_batch_size: len(test_labels)})
                print('Step(%d): %d, Loss(te): %f, accuracy(te): %f' % (j, loop_count, loss_val, acc_val))
                graph_y[loop_count] = acc_val
            nn.writer.add_summary(summary, loop_count)
            i += 1
            loop_count += 1
        # test_data
        print("------test_data------")
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss_test[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing,
                       nn.input_batch_size: len(test_labels)})
        print('Step: %d, Loss(te): %f, accuracy(te): %f' % (loop_count, loss_val, acc_val))
        """
        result is written in 0206_log_SAE.txt
        """
        f = open("0206_log_SAE.txt", 'a')
        f.write("%s, batchsize:%d, train_size:%d \n" % (args[1], batchsize, train_size))
        f.write('Step(%d): %d, Loss(te): %f, Accuracy: %f \n' % (layer_num-1, loop_count, loss_val, acc_val))
        f.close()

        j += 1
        loop_count += 1

    print("max_test_accuracy")

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
    # print("final_accuracy:{0}".format(final_accuracy_test))
    print("final_accuracy's_hidden_layer_num:{0}".format(final_loss_layer))
    print(loop_count)
    for i in range(early_stopping_num):
        # graph_y[loop_count + i] = None
        graph_train_y[loop_count + i] = None

    ax = plt.subplot()
    if layer_num == 1:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:2")
    if layer_num == 2:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:3")
    if layer_num == 3:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:4")
    if layer_num == 4:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:5")
    if layer_num == 5:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:6")
    if layer_num == 6:
        ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10, label="中間層数:7")
    #ax.plot(graph_x, graph_train_y[-i], linewidth=1.0, marker=".", markersize=10)
    ax.grid(which="both")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    # plt.title(str(train_size) + "images:MLP")
    ax.set_xlim([0, loop_count - 1 + last_early_stopping_num])

    np.save("SAE_y", graph_y)
    if dataset_flag == 0:
        np.save("SAE_mnist", graph_y)
    if dataset_flag == 1:
        np.save("SAE_fashion", graph_y)
    if dataset_flag == 2:
        np.save("SAE_cifar10", graph_y)
    if dataset_flag == 3:
        np.save("SAE_cifar100", graph_y)

    np.save("graph_x", graph_x)

    # plt.show()

    data_nonliner = np.load("nonliner_graph_y.npy")

    ax = plt.subplot()
    ax.plot(graph_x, graph_y, linewidth=1.0, marker=".", markersize=10)
    for i in range(layer_num + 1):
        ax.plot(graph_x, data_nonliner[-i], linewidth=1.0, marker=",", markersize=10)
    ax.grid(which="both")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.title(str(train_size) + "images:"+path.splitext(path.basename(__file__))[0])
    ax.set_xlim([0, loop_count - 1])
    # plt.show()

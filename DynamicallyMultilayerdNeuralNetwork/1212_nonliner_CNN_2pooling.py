import tensorflow as tf
import numpy as np
import random
import math
import os
import xlwt

# import load_notMNIST_1031

# np.random.seed(20160612)
# tf.set_random_seed(20160612)
np.random.seed(20171109)
tf.set_random_seed(20171109)

image_size = 28 * 28
output_size = 10
batchsize = 10
hidden_unit_number = 500
input_channel = 1
input_width = int(np.sqrt(image_size))
pooling_down = 4


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
                x = tf.placeholder(tf.float32, [None, input_size * input_channel])
                x_image = tf.reshape(x, [-1, int(input_width), int(input_width), int(input_channel)])

        with tf.name_scope('keep_prob'):
            keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('training_flag'):
            training = tf.placeholder(tf.bool)

        with tf.name_scope('filter_size'):
            with tf.name_scope('num_units8'):
                hidden_unit = hidden_unit_number
            with tf.name_scope('num_units8'):
                num_units8 = 64
            with tf.name_scope('num_units7'):
                num_units7 = 64
            with tf.name_scope('num_units6'):
                num_units6 = 64
                num_units5 = 128
                num_units4 = 128
                num_units3 = 256
                num_units2 = 256
                num_units1 = 512
                num_units0 = 512  # output_size

        with tf.name_scope('network_architecture'):
            with tf.name_scope('layer8'):
                W_conv8 = tf.Variable(
                    tf.random_uniform(shape=[7, 7, input_channel, num_units8],
                                      minval=-np.sqrt(6.0 / (3 * 3 * input_channel)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * input_channel)),
                                      dtype=tf.float32))
                # W_conv8 = tf.Variable(tf.truncated_normal([3, 3, input_channel, num_units8], stddev=0.01))
                b_conv8 = tf.Variable(tf.constant(0.1, shape=[num_units8]))
            with tf.name_scope('layer7'):
                W_conv7 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units8, num_units7],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units8)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units8)),
                                      dtype=tf.float32))
                # W_conv7 = tf.Variable(tf.truncated_normal([3, 3, num_units8, num_units7], stddev=0.01))
                b_conv7 = tf.Variable(tf.constant(0.1, shape=[num_units7]))
            with tf.name_scope('layer6'):
                W_conv6 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units7, num_units6],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units7)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units7)),
                                      dtype=tf.float32))
                # W_conv6 = tf.Variable(tf.truncated_normal([3, 3, num_units7, num_units6], stddev=0.01))
                b_conv6 = tf.Variable(tf.constant(0.1, shape=[num_units6]))
            with tf.name_scope('layer5'):
                W_conv5 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units6, num_units5],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units6)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units6)),
                                      dtype=tf.float32))
                # W_conv5 = tf.Variable(tf.truncated_normal([3, 3, num_units6, num_units5], stddev=0.01))
                b_conv5 = tf.Variable(tf.constant(0.1, shape=[num_units5]))
            with tf.name_scope('layer4'):
                W_conv4 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units5, num_units4],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units5)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units5)),
                                      dtype=tf.float32))
                # W_conv4 = tf.Variable(tf.truncated_normal([3, 3, num_units5, num_units4], stddev=0.01))
                b_conv4 = tf.Variable(tf.constant(0.1, shape=[num_units4]))
            with tf.name_scope('layer3'):
                W_conv3 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units4, num_units3],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units4)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units4)),
                                      dtype=tf.float32))
                # W_conv3 = tf.Variable(tf.truncated_normal([3, 3, num_units4, num_units3], stddev=0.01))
                b_conv3 = tf.Variable(tf.constant(0.1, shape=[num_units3]))
            with tf.name_scope('layer2'):
                W_conv2 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units3, num_units2],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units3)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units3)),
                                      dtype=tf.float32))
                # W_conv2 = tf.Variable(tf.truncated_normal([3, 3, num_units3, num_units2], stddev=0.01))
                b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
            with tf.name_scope('layer1'):
                W_conv1 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units2, num_units1],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units2)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units2)),
                                      dtype=tf.float32))
                # W_conv1 = tf.Variable(tf.truncated_normal([3, 3, num_units2, num_units1], stddev=0.01))
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_units1]))
            with tf.name_scope('layer0'):
                W_conv0 = tf.Variable(
                    tf.random_uniform(shape=[3, 3, num_units1, num_units0],
                                      minval=-np.sqrt(6.0 / (3 * 3 * num_units1)),
                                      maxval=np.sqrt(6.0 / (3 * 3 * num_units1)),
                                      dtype=tf.float32))
                # W_conv0 = tf.Variable(tf.truncated_normal([3, 3, num_units1, num_units0], stddev=0.01))
                b_conv0 = tf.Variable(tf.constant(0.1, shape=[num_units0]))
            with tf.name_scope('full_connected_layer'):
                w1 = tf.Variable(
                    tf.random_uniform(shape=[512, 1000],
                                      minval=-np.sqrt(6.0 / (512 * num_units0 + 1000)),
                                      maxval=np.sqrt(6.0 / (512 * num_units0 + 1000)),
                                      dtype=tf.float32))
                b1 = tf.Variable(tf.constant(0.1, shape=[1000]))
                w0 = tf.Variable(
                    tf.random_uniform(shape=[1000, output_size],
                                      minval=-np.sqrt(6.0 / (output_size + 1000)),
                                      maxval=np.sqrt(6.0 / (output_size + 1000)),
                                      dtype=tf.float32))
                b0 = tf.Variable(tf.zeros([output_size]))
            """
            with tf.name_scope('gate'):
                gate_9 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_8 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_7 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_6 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_5 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_4 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_3 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_2 = tf.Variable(tf.constant(1.0, shape=[1]))
                gate_1 = tf.Variable(tf.constant(1.0, shape=[1]))
            """
        with tf.name_scope('feed_forword'):
            with tf.name_scope('conv4'):
                h_pool_input = tf.nn.max_pool(x_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                h_conv8 = tf.nn.conv2d(h_pool_input, W_conv8, strides=[1, 1, 1, 1], padding='SAME')
                h_conv8_batch_normalized = tf.layers.batch_normalization(h_conv8, training=training)
                h_conv8_cutoff = tf.nn.relu(h_conv8_batch_normalized + b_conv8)

                h_pool_8 = tf.nn.max_pool(h_conv8_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv3'):
                h_conv7 = tf.nn.conv2d(h_pool_8, W_conv7, strides=[1, 1, 1, 1], padding='SAME')
                h_conv7_batch_normalized = tf.layers.batch_normalization(h_conv7, training=training)
                h_conv7_cutoff = tf.nn.relu(h_conv7_batch_normalized + b_conv7)

                h_conv6 = tf.nn.conv2d(h_conv7_cutoff, W_conv6, strides=[1, 1, 1, 1], padding='SAME')
                h_conv6_batch_normalized = tf.layers.batch_normalization(h_conv6, training=training)
                h_conv6_cutoff = tf.nn.relu(h_conv6_batch_normalized + b_conv6)

                h_pool_6 = tf.nn.max_pool(h_conv6_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv2'):
                h_conv5 = tf.nn.conv2d(h_pool_6, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
                h_conv5_batch_normalized = tf.layers.batch_normalization(h_conv5, training=training)
                h_conv5_cutoff = tf.nn.relu(h_conv5_batch_normalized + b_conv5)

                h_conv4 = tf.nn.conv2d(h_conv5_cutoff, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
                h_conv4_batch_normalized = tf.layers.batch_normalization(h_conv4, training=training)
                h_conv4_cutoff = tf.nn.relu(h_conv4_batch_normalized + b_conv4)

                h_pool_4 = tf.nn.max_pool(h_conv4_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv1'):
                h_conv3 = tf.nn.conv2d(h_pool_4, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
                h_conv3_batch_normalized = tf.layers.batch_normalization(h_conv3, training=training)
                h_conv3_cutoff = tf.nn.relu(h_conv3_batch_normalized + b_conv3)

                h_conv2 = tf.nn.conv2d(h_conv3_cutoff, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
                h_conv2_batch_normalized = tf.layers.batch_normalization(h_conv2, training=training)
                h_conv2_cutoff = tf.nn.relu(h_conv2_batch_normalized + b_conv2)

                h_pool_2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('conv0'):
                h_conv1 = tf.nn.conv2d(h_pool_2, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
                h_conv1_batch_normalized = tf.layers.batch_normalization(h_conv1, training=training)
                h_conv1_cutoff = tf.nn.relu(h_conv1_batch_normalized + b_conv1)

                h_conv0 = tf.nn.conv2d(h_conv1_cutoff, W_conv0, strides=[1, 1, 1, 1], padding='SAME')
                h_conv0_batch_normalized = tf.layers.batch_normalization(h_conv0, training=training)
                h_conv0_cutoff = tf.nn.relu(h_conv0_batch_normalized + b_conv0)

        with tf.name_scope('shortcut'):
            """
            h_pool_8_0 = tf.nn.max_pool(h_conv8_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_8_1 = tf.nn.max_pool(h_pool_8_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_8_2 = tf.nn.max_pool(h_pool_8_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_8_3 = tf.nn.max_pool(h_pool_8_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_8_0 = tf.reshape(h_pool_8_0, [-1, 512])

            h_pool_6_0 = tf.nn.max_pool(h_conv6_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_6_1 = tf.nn.max_pool(h_conv6_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_6_0 = tf.reshape(h_pool_6_0, [-1, 512])

            h_pool_4_0 = tf.nn.max_pool(h_conv4_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_4_0 = tf.reshape(h_pool_4_0, [-1, 512])

            h_pool_2_0 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_2_0 = tf.reshape(h_pool_2_0, [-1, 512])

            """
            h_pool_8_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_8_0 = tf.reshape(h_pool_8_0, [-1, 512])
            h_pool_6_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_6_0 = tf.reshape(h_pool_6_0, [-1, 512])
            h_pool_4_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_4_0 = tf.reshape(h_pool_4_0, [-1, 512])
            h_pool_2_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_2_0 = tf.reshape(h_pool_2_0, [-1, 512])
            h_pool_0_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_0_0 = tf.reshape(h_pool_0_0, [-1, 512])

            """
            h_pool_7 = tf.nn.max_pool(tf.nn.relu(h_conv7 +
                                                 h_conv8 +
                                                 b_conv7)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_7 = tf.reshape(h_pool_7, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_6 = tf.nn.max_pool(tf.nn.relu(h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv6)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_6 = tf.reshape(h_pool_6, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_5 = tf.nn.max_pool(tf.nn.relu(h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv5)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_5 = tf.reshape(h_pool_5, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_4 = tf.nn.max_pool(tf.nn.relu(h_conv4 +
                                                 h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv4)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_4 = tf.reshape(h_pool_4, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_3 = tf.nn.max_pool(tf.nn.relu(h_conv3 +
                                                 h_conv4 +
                                                 h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv3)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_3 = tf.reshape(h_pool_3, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_2 = tf.nn.max_pool(tf.nn.relu(h_conv2 +
                                                 h_conv3 +
                                                 h_conv4 +
                                                 h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv2)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_2 = tf.reshape(h_pool_2, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_1 = tf.nn.max_pool(tf.nn.relu(h_conv1 +
                                                 h_conv2 +
                                                 h_conv3 +
                                                 h_conv4 +
                                                 h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv1)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_1 = tf.reshape(h_pool_1, [-1, int((image_size / pooling_down) * num_units0)])

            h_pool_0 = tf.nn.max_pool(tf.nn.relu(h_conv0 +
                                                 h_conv1 +
                                                 h_conv2 +
                                                 h_conv3 +
                                                 h_conv4 +
                                                 h_conv5 +
                                                 h_conv6 +
                                                 h_conv7 +
                                                 h_conv8 +
                                                 b_conv0)
                                      , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_0 = tf.reshape(h_pool_0, [-1, int((image_size / pooling_down) * num_units0)])
            """
            """
            h_pool_8 = tf.nn.max_pool(h_conv8_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_8 = tf.reshape(h_pool_8, [-1, 14 * 14 * num_units0])

            h_pool_7 = tf.nn.max_pool(h_conv7_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_7 = tf.reshape(h_pool_7, [-1, 14 * 14 * num_units0])

            h_pool_6 = tf.nn.max_pool(h_conv6_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_6 = tf.reshape(h_pool_6, [-1, 14 * 14 * num_units0])

            h_pool_5 = tf.nn.max_pool(h_conv5_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_5 = tf.reshape(h_pool_5, [-1, 14 * 14 * num_units0])

            h_pool_4 = tf.nn.max_pool(h_conv4_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_4 = tf.reshape(h_pool_4, [-1, 14 * 14 * num_units0])

            h_pool_3 = tf.nn.max_pool(h_conv3_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_3 = tf.reshape(h_pool_3, [-1, 14 * 14 * num_units0])

            h_pool_2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_2 = tf.reshape(h_pool_2, [-1, 14 * 14 * num_units0])

            h_pool_1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_1 = tf.reshape(h_pool_1, [-1, 14 * 14 * num_units0])

            h_pool_0 = tf.nn.max_pool(h_conv0_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            h_pool_flat_0 = tf.reshape(h_pool_0, [-1, 14 * 14 * num_units0])
            """

            hidden_1_8 = tf.nn.relu(tf.matmul(h_pool_flat_8_0, w1) + b1)
            hidden_1_6 = tf.nn.relu(tf.matmul((h_pool_flat_6_0 + h_pool_flat_8_0) / 2, w1) + b1)
            hidden_1_4 = tf.nn.relu(tf.matmul((h_pool_flat_4_0 + h_pool_flat_6_0 + h_pool_flat_8_0) / 3, w1) + b1)
            hidden_1_2 = tf.nn.relu(
                tf.matmul((h_pool_flat_2_0 + h_pool_flat_4_0 + h_pool_flat_6_0 + h_pool_flat_8_0) / 4, w1) + b1)
            hidden_1_0 = tf.nn.relu(
                tf.matmul((h_pool_flat_0_0 + h_pool_flat_2_0 + h_pool_flat_4_0 + h_pool_flat_6_0 + h_pool_flat_8_0) / 5,
                          w1) + b1)

            """
            hidden_1_8 = tf.nn.relu(tf.matmul(h_pool_flat_8, w1) + b1)
            hidden_1_7 = tf.nn.relu(tf.matmul(h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_6 = tf.nn.relu(tf.matmul(h_pool_flat_6 + h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_5 = tf.nn.relu(tf.matmul(h_pool_flat_5 + h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_4 = tf.nn.relu(tf.matmul(h_pool_flat_4 + h_pool_flat_5 + h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_3 = tf.nn.relu(tf.matmul(h_pool_flat_3 + h_pool_flat_4 + h_pool_flat_5 + h_pool_flat_7 +
                                              h_pool_flat_8, w1) + b1)
            hidden_1_2 = tf.nn.relu(tf.matmul(h_pool_flat_2 + h_pool_flat_3 + h_pool_flat_4 + h_pool_flat_5 +
                                              h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_1 = tf.nn.relu(tf.matmul(h_pool_flat_1 + h_pool_flat_2 + h_pool_flat_3 + h_pool_flat_4 +
                                              h_pool_flat_5 + h_pool_flat_7 + h_pool_flat_8, w1) + b1)
            hidden_1_0 = tf.nn.relu(tf.matmul(h_pool_flat_0 + h_pool_flat_1 + h_pool_flat_2 + h_pool_flat_3 +
                                              h_pool_flat_4 + h_pool_flat_5 + h_pool_flat_7 + h_pool_flat_8, w1) + b1)

            """
            with tf.name_scope('p0'):
                p0 = tf.nn.softmax(tf.matmul(hidden_1_8, w0) + b0)
            with tf.name_scope('p1'):
                p1 = tf.nn.softmax(tf.matmul(hidden_1_6, w0) + b0)
            with tf.name_scope('p2'):
                p2 = tf.nn.softmax(tf.matmul(hidden_1_4, w0) + b0)
            with tf.name_scope('p3'):
                p3 = tf.nn.softmax(tf.matmul(hidden_1_2, w0) + b0)
            with tf.name_scope('p4'):
                p4 = tf.nn.softmax(tf.matmul(hidden_1_0, w0) + b0)

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
            # learning_rate = 0.001  # 0.0001
            # AdamOptimizer
            """
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
            """
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
            with tf.name_scope('train_step0'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    """
                    train_step0 = tf.train.AdamOptimizer().minimize(loss0, var_list=[W_conv8, b_conv8, w1, b1, w0, b0])
                    train_step1 = tf.train.AdamOptimizer().minimize(loss1, var_list=[W_conv7, b_conv7, w1, b1, w0, b0])
                    train_step2 = tf.train.AdamOptimizer().minimize(loss2, var_list=[W_conv6, b_conv6, w1, b1, w0, b0])
                    train_step3 = tf.train.AdamOptimizer().minimize(loss3, var_list=[W_conv5, b_conv5, w1, b1, w0, b0])
                    train_step4 = tf.train.AdamOptimizer().minimize(loss4, var_list=[W_conv4, b_conv4, w1, b1, w0, b0])
                    """
                    train_step0 = tf.train.AdamOptimizer().minimize(loss0)
                    train_step1 = tf.train.AdamOptimizer().minimize(loss1)
                    train_step2 = tf.train.AdamOptimizer().minimize(loss2)
                    train_step3 = tf.train.AdamOptimizer().minimize(loss3)
                    train_step4 = tf.train.AdamOptimizer().minimize(loss4)
            # Adam
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
        tf.summary.scalar("total_accuracy", tf.reduce_sum(total_accuracy))
        tf.summary.scalar("train_total_accuracy", tf.reduce_sum(train_total_accuracy))
        tf.summary.scalar("total_loss", tf.reduce_sum(total_loss))
        tf.summary.scalar("train_total_loss", tf.reduce_sum(train_total_loss))
        tf.summary.scalar("accuracy", tf.reduce_sum(total_accuracy))

        self.x, self.t = x, t
        self.loss = loss
        self.accuracy = accuracy
        self.total_accuracy = total_accuracy
        self.total_loss = total_loss
        self.train_total_loss = train_total_loss
        self.train_step = train_step

        self.keep_prob = keep_prob

        self.assign_op_total_accuracy = assign_op_total_accuracy
        self.input_placeholder_total_accuracy = input_placeholder_total_accuracy

        self.assign_op_train_total_accuracy = assign_op_train_total_accuracy
        self.input_placeholder_train_total_accuracy = input_placeholder_train_total_accuracy

        self.assign_op_train_total_loss = assign_op_train_total_loss
        self.input_placeholder_train_total_loss = input_placeholder_train_total_loss

        self.assign_op_total_loss = assign_op_total_loss
        self.input_placeholder_total_loss = input_placeholder_total_loss

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

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./log/1105_MNIST_nonliner", sess.graph)
        writer = tf.summary.FileWriter("./log/1212_nonliner_CNN_batch_normalized", sess.graph)
        # ここでログファイルを保存するディレクトリとファイル名を決定する

        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = tf.train.Saver(max_to_keep=None)


def initialize_MNIST():
    global batchsize
    batchsize = 32
    global image_size
    image_size = 28 * 28  # 32 * 32
    global input_channel
    input_channel = 1  # 3
    global hidden_unit_number
    hidden_unit_number = 16
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4


def initialize_cifar():
    global batchsize
    batchsize = 256
    global image_size
    image_size = 32 * 32  # 28*28
    global input_channel
    input_channel = 3  # 1
    global hidden_unit_number
    hidden_unit_number = 16
    global input_width
    input_width = np.sqrt(image_size)
    global pooling_down
    pooling_down = 4


if __name__ == '__main__':
    initialize_MNIST()
    train_size = 55000  # 10000 is good
    layer_num = 0
    # start = layer_num
    start = 0
    early_stopping_num = 20
    session_name = "./session_log/saver_1212_nonliner_CNN_batch_normalized"
    training = True
    testing = False

    nn = layer()
    batch_xs = np.mat([[0.0 for n in range(image_size * input_channel)] for k in range(batchsize)])
    batch_ts = np.mat([[0.0 for n in range(output_size)] for k in range(batchsize)])
    """
    data = np.load("MNIST_train_data.npy")[0:train_size]
    labels = np.load("MNIST_train_labels.npy")[0:train_size]
    test_data = np.load("MNIST_test_data.npy")
    test_labels = np.load("MNIST_test_labels.npy")
    valid_data = np.load("MNIST_valid_data.npy")[0:int(train_size / 10)]
    valid_labels = np.load("MNIST_valid_labels.npy")[0:int(train_size / 10)]
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
    """
    print(data.shape)

    train_test_size = train_size
    if train_size > 2000:
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

    loop_len = 100000  # 400000
    print("loop_len:{0}".format(loop_len))
    drop_out_rate = 1.0
    loop_count = 0
    acc_val = 0.
    gate_value = [1., 1., 1., 1.]
    print("dropout:{0}".format(drop_out_rate))
    train_accracy = 0.0
    best_loss = 10000.
    stopping_step = 0

    for j in range(start, layer_num + 1):
        # test_data
        print("------test_data------")
        summary, loss_val_valid, acc_val_valid = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing})
        print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_valid, acc_val_valid))
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))

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
                                                         nn.training: training})

            tmp_loss = nn.sess.run(nn.loss[j], feed_dict={nn.x: valid_data, nn.t: valid_labels,
                                                          nn.keep_prob: drop_out_rate, nn.training: testing})
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
                """
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: train_test_data, nn.t: train_test_labels, nn.keep_prob: 1.0, nn.training: testing})
                # print('Step(%d): %d, Loss(tr): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                nn.sess.run(nn.assign_op_train_total_accuracy,
                            feed_dict={nn.input_placeholder_train_total_accuracy: acc_val, nn.training: testing})
                train_loss = loss_val
                train_accracy = acc_val
                nn.sess.run(nn.assign_op_train_total_loss, feed_dict={nn.input_placeholder_train_total_loss: loss_val, nn.training: testing})
                train_test_output.append(acc_val)
                """
                # valid_data
                loss_val, acc_val = nn.sess.run(
                    [nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing})
                print('Step(%d): %d, Loss(va): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))

                # test_data
                summary, loss_val, acc_val = nn.sess.run(
                    [nn.summary, nn.loss[j], nn.accuracy[j]],
                    feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing})
                # print('Step(%d): %d, Loss(te): %f, Accuracy: %f' % (j, loop_count, loss_val, acc_val))
                test_output.append(acc_val)
            nn.writer.add_summary(summary, loop_count)
            i += 1
            loop_count += 1
        # test_data
        print("------test_data------")
        summary, loss_val_valid, acc_val_valid = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: valid_data, nn.t: valid_labels, nn.keep_prob: 1.0, nn.training: testing})
        print('Step: %d, Loss(va): %f, Accuracy: %f' % (loop_count, loss_val_valid, acc_val_valid))
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss[j], nn.accuracy[j]],
            feed_dict={nn.x: test_data, nn.t: test_labels, nn.keep_prob: 1.0, nn.training: testing})
        print('Step: %d, Loss(te): %f, Accuracy: %f' % (loop_count, loss_val, acc_val))
        if acc_val_valid > final_acc_valid:
            final_acc_valid = acc_val_valid
            final_accuracy_test = acc_val
            final_loss_layer = j
        print("------test__end------")
        # gate_value[j] = train_accracy
        j += 1
        loop_count += 1
    print("max_test_accuracy")
    print("te {0}:{1}".format(max(test_output),
                              [i * 100 for i, x in enumerate(test_output) if x == max(test_output)]))
    """
    print("max_train_accuracy")
    print("tr {0}:{1}".format(max(train_test_output), [i * 100 for i, x in enumerate(train_test_output) if
                                                       x == max(train_test_output)]))
    print("train_size:{0}".format(train_size))
    """

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

#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras import metrics
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers import Input, Dense, Reshape

import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)

### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

from keras.utils import np_utils

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)
    return x


def lambda_output(input_shape):
    return input_shape[:2]

def minibatch_discrimination(d_out):
    ### Minibatch Discrimination用のパラメータ
    num_kernels = 15  # 100まで大きくすると識別機誤差が0.5で固定
    dim_per_kernel = 50
    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)
    ### Minibatch Discrimination用のパラメータ

    x_mbd_true = M(d_out)
    x_mbd_true = Reshape((num_kernels, dim_per_kernel))(x_mbd_true)
    x_mbd_true = MBD(x_mbd_true)
    d_out = keras.layers.concatenate([d_out, x_mbd_true])
    return d_out

def weightGAN_Model(input_size=4, wSize=20, output_size=3, use_mbd=False):
    ### 生成器定義
    _g_dense1 = Dense(wSize, activation='sigmoid', name='g_dense1')
    _g_dense2 = Dense(output_size, activation='softmax', name='x_out')
    ### 生成器定義

    ### 識別機定義
    _d_dense1 = Dense(100, activation='relu', name='d_dense1')
    _d_out = Dense(1, activation='sigmoid', name='d_out')
    ### 識別機定義

    ### 生成器の順伝播　[入力:inputs_z(入力画像) 出力:x(クラス分類結果)]
    inputs_z = Input(shape=(input_size,), name='Z')  # 入力を取得
    g_dense1 = _g_dense1(inputs_z)
    x = _g_dense2(g_dense1)
    ### 生成器の順伝播　

    ### 識別器の順伝播
    inputs_labels = Input(shape=(output_size,), name='label')  # 入力を取得
    # 識別器の順伝播（真入力）　[入力:[inputs_label, inputs_w] 出力:d_out_true]　# true画像の入力の時(Gの出力を外部に吐き出し、それを入力すると勾配計算がされない)
    inputs_w = Input(shape=(wSize,), name='weight')  # 入力重みを取得
    d_out_true = _d_dense1(keras.layers.concatenate([inputs_w, inputs_labels]))  # d_dense1(inputs_w)
    if use_mbd:
        d_out_true = minibatch_discrimination(d_out_true)
    d_out_true = _d_out(d_out_true)
    # 識別器の順伝播（真入力）

    # 識別器の順伝播（偽入力）　[入力:[[inputs_label, g_dense1(生成器の中間出力)], inputs_w] 出力:d_out_fake]
    d_out_fake = _d_dense1(keras.layers.concatenate([g_dense1, inputs_labels]))  # d_dense1(g_dense1)
    if use_mbd:
        d_out_fake = minibatch_discrimination(d_out_fake)
    d_out_fake = _d_out(d_out_fake)
    # 識別器の順伝播（偽入力）
    ### 識別器の順伝播

    ### モデル定義
    g = Model(inputs=[inputs_z], outputs=[x, g_dense1], name='G')
    d = Model(inputs=[inputs_w, inputs_labels], outputs=[d_out_true], name='D')
    c = Model(inputs=[inputs_z, inputs_labels], outputs=[x, d_out_fake], name='C')  # end-to-end学習(g+d)
    classify = Model(inputs=[inputs_z], outputs=[x], name='classify')
    classify.compile(loss='categorical_crossentropy',
                     optimizer="adam",
                     metrics=[metrics.categorical_accuracy])
    for layer in d.layers:  # 生成器の学習時は識別機は固定
        layer.trainable = False
    c_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    c.compile(optimizer=c_opt,
              loss={'x_out': 'categorical_crossentropy', 'd_out': 'mse'},
              loss_weights={'x_out': 1., 'd_out': 0.})
    for layer in d.layers:
        layer.trainable = True
    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d.compile(optimizer=d_opt, loss='mean_squared_error')
    ### モデル定義

    ### モデル構造を出力
    g.summary()
    d.summary()
    c.summary()
    classify.summary()
    ### モデル構造を出力

    return g, d, c, classify
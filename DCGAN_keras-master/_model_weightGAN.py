#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras import metrics, regularizers
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers import Input, Dense, Reshape, multiply

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
from tensorflow_model_optimization.sparsity import keras as sparsity
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=10000,
                                                   frequency=100)
}
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
    activation = "relu"
    _g_dense1 = Dense(wSize, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense1_')
    _g_dense2 = Dense(wSize//2, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense2_')
    _g_dense3 = Dense(wSize//4, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense3_')
    _g_dense4 = Dense(wSize//8, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense4_')
    _g_dense_output = Dense(output_size, activation='softmax', name='x_out')
    ### 生成器定義

    ### 識別機定義
    _d_dense1 = Dense(100, activation='relu', name='d_dense1')
    _d_out = Dense(1, activation='sigmoid', name='d_out')
    ### 識別機定義

    ### 生成器の順伝播　[入力:inputs_z(入力画像) 出力:x(クラス分類結果)]
    inputs_z = Input(shape=(input_size,), name='Z')  # 入力を取得
    g_mask_1 = Input(shape=(wSize,), name='g_mask_1')

    g_dense1 = _g_dense1(inputs_z)
    g_dense1 = multiply([g_dense1, g_mask_1])
    g_dense2 = _g_dense2(g_dense1)
    g_dense3 = _g_dense3(g_dense2)
    g_dense4 = _g_dense4(g_dense3)
    x = _g_dense_output(g_dense4)
    ### 生成器の順伝播　

    ### 1 vs その他クラス分類
    _g_dense_binary_class_output = Dense(2, activation='softmax', name='binary_class_out')
    binary_class_output = _g_dense_binary_class_output(g_dense4)
    ### 1 vs その他クラス分類

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
    g = Model(inputs=[inputs_z, g_mask_1], outputs=[x, g_dense1], name='G')
    d = Model(inputs=[inputs_w, inputs_labels], outputs=[d_out_true], name='D')
    c = Model(inputs=[inputs_z, inputs_labels, g_mask_1], outputs=[x, d_out_fake], name='C')  # end-to-end学習(g+d)
    classify = Model(inputs=[inputs_z, g_mask_1], outputs=[x], name='classify')
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

    binary_classify = Model(inputs=[inputs_z, g_mask_1], outputs=[binary_class_output], name='binary_classify')
    binary_classify.compile(loss='categorical_crossentropy',
                     optimizer="adam",
                     metrics=[metrics.categorical_accuracy])

    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense1 = Model(inputs=[inputs_z, g_mask_1], outputs=[g_dense1], name='g_dense1')
    G_dense1.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense2 = Model(inputs=[inputs_z, g_mask_1], outputs=[g_dense2], name='g_dense2')
    G_dense2.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense3 = Model(inputs=[inputs_z, g_mask_1], outputs=[g_dense3], name='g_dense3')
    G_dense3.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense4 = Model(inputs=[inputs_z, g_mask_1], outputs=[g_dense4], name='g_dense4')
    G_dense4.compile(optimizer=d_opt, loss='mean_squared_error')
    G_output = Model(inputs=[inputs_z, g_mask_1], outputs=[x], name='g_dense_out')
    G_output.compile(optimizer=d_opt, loss='mean_squared_error')

    for layer in G_dense1.layers:
        print(layer.name)
        # layer.trainable = False
    # exit()
    freezed_classify_1 = Model(inputs=[inputs_z, g_mask_1], outputs=[x], name='freezed_classify_1')
    freezed_classify_1.compile(loss='categorical_crossentropy',
                     optimizer="adam",
                     metrics=[metrics.categorical_accuracy])
    ### モデル定義

    ### モデル構造を出力
    print("g.summary()")
    g.summary()
    print("d.summary()")
    d.summary()
    print("c.summary()")
    c.summary()
    print("classify.summary()")
    classify.summary()
    hidden_layers=[G_dense1, G_dense2, G_dense3, G_dense4, G_output]
    for i in range(len(hidden_layers)):
        print("hiddden_layers[{}].summary()".format(i))
        hidden_layers[i].summary()
    freezed_classify_1.summary()
    return g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1

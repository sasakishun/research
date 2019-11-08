#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras import metrics, regularizers
from keras import activations, regularizers, initializers, constraints
from keras.layers.core import Lambda, Activation
from keras.models import Model
from keras.layers import Input, Dense, Reshape, multiply, Dropout, BatchNormalization
from keras.layers.normalization import BatchNormalization
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)

### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import MyDenseLayer
from MyLayer import *
from _tree_functions import *

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

from keras.utils import np_utils

# from tensorflow_model_optimization.sparsity import keras as sparsity
"""
pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=10000,
                                                   frequency=100)
}
"""


def relu(a):
    return np.where(a < 0, 0, a)


def softmax(a):
    # 一番大きい値を取得
    c = np.max(a)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a
    return y


# softmax層出力(A, B, C)=outputsから、softmax層への入力(a, b, c)=inputsを逆算
# a,b,cの内一つ以外指定し、その一つの入力を計算する
def arg_softmax(inputs_removed_target, output):
    return np.log(sum([np.exp(i) for i in inputs_removed_target]) * output / (1 - output))


def arg_relu(y):
    return y


def arg_bn(output, bn_param, show_parameter=True):
    if show_parameter:
        print("\nbn_param:{}\n".format({"beta": bn_param["beta"],
                                        "mean": bn_param["mean"],
                                        "gamma": bn_param["gamma"],
                                        "var": bn_param["var"],
                                        "epsilon": bn_param["epsilon"]}))
    # gamma == 0の場合に0除算発生->出力がnan
    if bn_param["gamma"] == 0:
        print("output:{}".format(output))
        return 0
    return ((output - bn_param["beta"]) * np.sqrt(bn_param["var"] + bn_param["epsilon"])) \
           / bn_param["gamma"] + bn_param["mean"]


def bn(x, bn_param, show_parameter=False):
    if show_parameter:
        print("\nbn_param:{}\n".format({"beta": bn_param["beta"],
                                        "mean": bn_param["mean"],
                                        "gamma": bn_param["gamma"],
                                        "var": bn_param["var"],
                                        "epsilon": bn_param["epsilon"]}))
    return ((x - bn_param["mean"]) / np.sqrt(bn_param["var"] + bn_param["epsilon"])) \
           * bn_param["gamma"] + bn_param["beta"]


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


def weightGAN_Model(input_size=4, wSize=20, output_size=3, use_mbd=False, dense_size=[]):
    ### 生成器定義
    activation = "relu"
    """
    if not binary_weights_setted:
        _g_dense1 = Dense(wSize, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense1_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense2 = Dense(max(wSize//2 ,2), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense2_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense3 = Dense(max(wSize//4 ,2), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense3_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense4 = Dense(max(wSize//8 ,2), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense4_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    else:
        _g_dense1 = Dense(wSize, activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense1_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense2 = Dense(max(wSize//2, 100), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense2_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense3 = Dense(max(wSize//4, 50), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense3_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        _g_dense4 = Dense(max(wSize//8, 20), activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense4_',
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    """
    _g_dense1 = Dense(dense_size[1], activation=activation, kernel_regularizer=regularizers.l1(0.01), name='g_dense1_',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    _g_dense2 = Dense(dense_size[2], activation=activation, kernel_regularizer=regularizers.l1(0.01),
                      name='g_dense2_',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    _g_dense3 = Dense(dense_size[3], activation=activation, kernel_regularizer=regularizers.l1(0.01),
                      name='g_dense3_',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    _g_dense4 = Dense(dense_size[4], activation=activation, kernel_regularizer=regularizers.l1(0.01),
                      name='g_dense4_',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
    _g_dense_output = Dense(output_size, activation='softmax', name='x_out',
                            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10, seed=None))
    ### 生成器定義

    ### 識別機定義
    _d_dense1 = Dense(100, activation='relu', name='d_dense1')
    _d_out = Dense(1, activation='sigmoid', name='d_out')
    ### 識別機定義

    ### 生成器の順伝播　[入力:inputs_z(入力画像) 出力:x(クラス分類結果)]
    g_mask_0 = Input(shape=(dense_size[0],), name='g_mask_0')
    g_mask_1 = Input(shape=(dense_size[1],), name='g_mask_1')
    g_mask_2 = Input(shape=(dense_size[2],), name='g_mask_2')
    g_mask_3 = Input(shape=(dense_size[3],), name='g_mask_3')
    g_mask_4 = Input(shape=(dense_size[4],), name='g_mask_4')
    g_masks = [g_mask_0, g_mask_1, g_mask_2, g_mask_3, g_mask_4]

    inputs_z = Input(shape=(dense_size[0],), name='Z')  # 入力を取得
    # inputs_z = multiply([inputs_z, g_mask_0]) # inputの値を変更するのはエラー発生
    g_dense1 = _g_dense1(inputs_z)
    g_dense1 = multiply([g_dense1, g_mask_1])
    # g_dense1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(g_dense1)
    g_dense2 = _g_dense2(g_dense1)
    g_dense2 = multiply([g_dense2, g_mask_2])
    # g_dense2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(g_dense2)
    g_dense3 = _g_dense3(g_dense2)
    g_dense3 = multiply([g_dense3, g_mask_3])
    # g_dense3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(g_dense3)
    g_dense4 = _g_dense4(g_dense3)
    g_dense4 = multiply([g_dense4, g_mask_4])
    # g_dense4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(g_dense4)
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
    for i in range(len([inputs_z] + g_masks)):
        print("([inputs_z] + g_masks)[{}]:{}".format(i, ([inputs_z] + g_masks)[i]))
    for i in range(len(dense_size)):
        print("dense_size[{}]:{}".format(i, dense_size[i]))
    for i in range(len(g_masks)):
        print("g_masks[{}]:{}".format(i, g_masks[i]))

    g = Model(inputs=[inputs_z] + g_masks, outputs=[x, g_dense1], name='G')
    d = Model(inputs=[inputs_w, inputs_labels], outputs=[d_out_true], name='D')
    c = Model(inputs=[inputs_z, inputs_labels] + g_masks, outputs=[x, d_out_fake], name='C')  # end-to-end学習(g+d)
    classify = Model(inputs=[inputs_z] + g_masks, outputs=[x], name='classify')
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

    binary_classify = Model(inputs=[inputs_z] + g_masks, outputs=[binary_class_output], name='binary_classify')
    binary_classify.compile(loss='categorical_crossentropy',
                            optimizer="adam",
                            metrics=[metrics.categorical_accuracy])

    d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense1 = Model(inputs=[inputs_z] + g_masks, outputs=[g_dense1], name='g_dense1')
    G_dense1.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense2 = Model(inputs=[inputs_z] + g_masks, outputs=[g_dense2], name='g_dense2')
    G_dense2.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense3 = Model(inputs=[inputs_z] + g_masks, outputs=[g_dense3], name='g_dense3')
    G_dense3.compile(optimizer=d_opt, loss='mean_squared_error')
    G_dense4 = Model(inputs=[inputs_z] + g_masks, outputs=[g_dense4], name='g_dense4')
    G_dense4.compile(optimizer=d_opt, loss='mean_squared_error')
    G_output = Model(inputs=[inputs_z] + g_masks, outputs=[x], name='g_dense_out')
    G_output.compile(optimizer=d_opt, loss='mean_squared_error')

    for layer in G_dense1.layers:
        # print(layer.name)
        layer.trainable = False
    # exit()
    freezed_classify_1 = Model(inputs=[inputs_z] + g_masks, outputs=[x], name='freezed_classify_1')
    freezed_classify_1.compile(loss='categorical_crossentropy',
                               optimizer="adam",
                               metrics=[metrics.categorical_accuracy])
    ### モデル定義

    ### モデル構造を出力
    """
    print("g.summary()")
    g.summary()
    print("d.summary()")
    d.summary()
    print("c.summary()")
    c.summary()
    print("classify.summary()")
    classify.summary()
    """
    hidden_layers = [G_dense1, G_dense2, G_dense3, G_dense4, G_output]
    # for i in range(len(hidden_layers)):
    # print("hiddden_layers[{}].summary()".format(i))
    # hidden_layers[i].summary()
    # freezed_classify_1.summary()
    return g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1


def get_hidden_nodes_num(input_size, all_combination_flag):
    # layer_num = 0
    hidden_nodes_num = []
    while input_size / 2 >= 1:
        hidden_nodes_num.append(input_size // 2)
        input_size = input_size // 2 + input_size % 2
    return hidden_nodes_num  # [input_size//(2**(i+1)) for i in range(layer_num)]


def tree(input_size, output_size, get_hidden_flag=False, all_combination_flag=False):
    activation = "relu"
    hidden_nodes_num = get_hidden_nodes_num(input_size, all_combination_flag)
    if get_hidden_flag:
        return hidden_nodes_num
    layer_num = len(hidden_nodes_num)
    print("layer_num:{}".format(layer_num))
    print("hidden_nodes_num:{}".format(hidden_nodes_num))
    ### モデル入力定義
    inputs = [Input(shape=(1,), name='inputs_{}'.format(i)) for i in range(input_size)]
    # calculated = [Lambda(lambda x: K.sqrt(x + 1.0), output_shape=(1,))(_inputs) for _inputs in inputs]
    for i in range(len(inputs)):
        print("inputs[{}]:{}".format(i, inputs[i]))
    ### モデル入力定義

    ### 中間層定義 入力列の隣合う要素同士を木の入力とする
    _dense = [[[Dense(1, activation=activation if i != layer_num - 1 else None,
                      kernel_regularizer=regularizers.l1(0.01),
                      name='dense{}_{}_{}'.format(output, i, j),
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
                for j in range(hidden_nodes_num[i])]
               for i in range(layer_num)]
              for output in range(output_size)]
    ### 中間層定義

    dense = [[inputs] for _ in range(output_size)]

    for _out in range(output_size):
        for i in range(layer_num):
            # print("i:{}".format(i))
            # print("dense[{}][{}]:{} type:{}".format(_out, i, dense[_out][i], type(dense[_out][i])))
            # print("dense[{}][{}]:{} type:{}".format(_out, i, len(dense[_out][i]), type(dense[_out][i])))
            odd_flag = False
            if len(dense[_out][i]) % 2 == 1:
                odd_node = dense[_out][i][-1]
                odd_flag = True
            ### dense[_out][i]の各要素を次層入力のためにconcatenate
            dense[_out][i] = [keras.layers.concatenate([dense[_out][i][j * 2], dense[_out][i][j * 2 + 1]])
                              for j in range(len(dense[_out][i]) // 2)]
            # dense[_out][i] = [Dropout(rate=0.1)(dense[_out][i][j]) for j in range(hidden_nodes_num[i])]

            ### dense[_out][i+1]に入力を伝播
            dense[_out].append([_dense[_out][i][j](dense[_out][i][j])
                                for j in range(hidden_nodes_num[i])])
            if odd_flag:
                dense[_out][-1].append(odd_node)
                # print("\nadd odd\n")
    output = keras.layers.concatenate([dense[i][-1][0] for i in range(len(dense))]) if len(dense) > 1 else dense[0][-1][
        0]
    output = keras.layers.Activation("softmax")(output)
    dense_tree = Model(inputs=inputs, outputs=output, name='dense_tree')
    dense_tree.compile(loss='categorical_crossentropy',
                       optimizer="adam",
                       metrics=[metrics.categorical_accuracy])
    return dense_tree


def mlp(input_size, hidden_size, output_size):
    activation = "relu"
    _dense = [
        Dense(hidden_size[j], activation=activation if j != 0 else None, kernel_regularizer=regularizers.l1(0.01),
              name='dense{}'.format(j),
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        for j in range(len(hidden_size))]
    _dense.append(Dense(output_size, activation="softmax", kernel_regularizer=regularizers.l1(0.01),
                        name='dense{}'.format(len(_dense)),
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
    inputs = Input(shape=(input_size,), name='inputs')
    dense = [inputs]
    for i in range(len(_dense)):
        dense.append(_dense[i](dense[i]))
    _mlp = Model(inputs=inputs, outputs=dense[-1], name='dense_tree')
    _mlp.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=[metrics.categorical_accuracy])
    return _mlp


def masked_mlp(input_size, hidden_size, output_size):
    # wineデータセットでは
    # input_size  : 13
    # hidden_size : [48, 24, 12, 6]
    # output_size : 3
    model_shape = [input_size] + hidden_size + [output_size]
    activation = "relu"
    _dense = [
        Dense(hidden_size[j], activation=activation if j != 0 else None, kernel_regularizer=regularizers.l1(0.01),
              name='dense{}'.format(j),
              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        for j in range(len(hidden_size))]
    _dense.append(Dense(output_size, activation="softmax", kernel_regularizer=regularizers.l1(0.01),
                        name='dense{}'.format(len(_dense)),
                        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))

    inputs = Input(shape=(input_size,), name='inputs')
    # 入力層から出力直前層の出力にmaskをmultiplyする
    mask = [Input(shape=(model_shape[i],), name='mask_{}'.format(i))
            for i in range(len(model_shape))]
    dense = [multiply([inputs, mask[0]])]
    for i in range(len(_dense)):
        dense.append(_dense[i](dense[i]))
        dense[-1] = multiply([dense[-1], mask[i + 1]])
    _mlp = Model(inputs=[inputs] + mask, outputs=dense[-1], name='dense_tree')
    _mlp.compile(loss='categorical_crossentropy',
                 optimizer="adam",
                 metrics=[metrics.categorical_accuracy])
    return _mlp


def _myMLP(model_shape, kernel_mask=None, bias_mask=None, trainable=True, set_weights=None):
    activation = "relu"
    input_size = model_shape[0]
    hidden_size = model_shape[1:-1]
    output_size = model_shape[-1]
    print("input:{}".format(input_size))
    print("hidden_size:{}".format(hidden_size))
    print("output_size:{}".format(output_size))
    _dense = [
        MyLayer(hidden_size[j], activation=activation,  # None,
                kernel_regularizer=regularizers.l1(0.01), name='dense{}'.format(j),
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
        for j in range(len(hidden_size))
    ]
    _dense.append(MyLayer(output_size, activation="softmax",
                          kernel_regularizer=regularizers.l1(0.01), name='dense{}'.format(len(_dense)),
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                          ))
    inputs = Input(shape=(input_size,), name='inputs')
    dense = [inputs]

    for i in range(len(_dense)):
        dense[i] = BatchNormalization(name="BN{}".format(i))(dense[i])
        dense.append(_dense[i](dense[i],
                               kernel_mask=kernel_mask[i] if kernel_mask is not None else None,
                               bias_mask=bias_mask[i] if bias_mask is not None else None)
                     )
        # if i < len(_dense) - 1:
        # dense[i + 1] = BatchNormalization(trainable=False)(dense[i + 1])
        # dense[i+1] = Activation(activation)(dense[i+1])
        # dense[i+1] = Dropout(rate=0.0001)(dense[i+1])

    mlp = Model(inputs=inputs, outputs=dense[-1], name='dense_tree')
    mlp.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=[metrics.categorical_accuracy])
    if set_weights:
        mlp.set_weights(set_weights)
    return mlp


def myMLP(model_shape, kernel_mask=None, bias_mask=None, trainable=True, set_weights=None):
    mlp = keras.models.Sequential()
    for layer_number in range(len(model_shape) - 1):
        input_size = model_shape[layer_number]
        hidden_size = model_shape[layer_number + 1]
        # print("layer_number:{} hidden_size:{}".format(layer_number, hidden_size))
        if layer_number + 1 == len(model_shape) - 1:  # 出力層
            activation = "softmax"
        else:
            activation = "relu"
        mlp.add(dense_layer_model(hidden_size, input_size, activation=activation, layer_number=layer_number,
                                  kernel_mask=kernel_mask[layer_number] if kernel_mask is not None else None,
                                  bias_mask=bias_mask[layer_number] if bias_mask is not None else None))
    mlp.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=[metrics.categorical_accuracy])
    print("weights:{}".format([np.shape(i) for i in mlp.get_weights()]))
    if set_weights:
        mlp.set_weights(set_weights)
    return mlp


def dense_layer_model(hidden_size, input_size, activation=None, layer_number=None, kernel_mask=None, bias_mask=None,
                      trainable=True):
    _dense = MyLayer(hidden_size, activation=activation,
                     kernel_regularizer=regularizers.l1(0.01), name='dense{}'.format(layer_number),
                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None))

    inputs = Input(shape=(input_size,), name='inputs')
    dense = BatchNormalization(name="BN_{}".format(layer_number),
                               beta_initializer='ones',
                               gamma_initializer='zeros',
                               moving_mean_initializer='zeros',
                               moving_variance_initializer='zeros',
                               )(inputs)
    dense = _dense(dense, kernel_mask=kernel_mask, bias_mask=bias_mask)

    dense_model = Model(inputs=inputs, outputs=dense, name='dense_layer_{}'.format(layer_number))
    return dense_model


def tree_mlp(input_size, output_size, kernel_mask=None, bias_mask=None, child_num=2, is_image=False):
    from _tree_functions import calculate_tree_shape
    print("input:{}".format(input_size))
    print("output_size:{}".format(output_size))
    hidden_size = calculate_tree_shape(input_size, output_size, child_num=child_num, is_image=is_image)
    print("tree_shape:{}".format([i for i in hidden_size]))
    if kernel_mask is None:
        # [13->21->12->6->3]のmlpのmaskを作成
        kernel_mask = get_tree_kernel_mask(hidden_size, child_num=child_num, is_image=is_image)
    # visualize_network(kernel_mask, )
    return myMLP(hidden_size, kernel_mask, bias_mask)


def get_tree_kernel_mask(hidden_size, child_num=2, show_mask=False, is_image=False):
    import math
    kernel_mask = [np.zeros((hidden_size[i - 1], hidden_size[i])) for i in range(1, len(hidden_size))]
    output_size = hidden_size[-1]
    if not is_image:
        for i in range(len(kernel_mask)):
            for j in range(hidden_size[i]):
                if i == 0:
                    for _class in range(output_size):
                        kernel_mask[i][j][j // child_num + (hidden_size[i + 1] // output_size) * _class] = 1
                else:
                    _class = j // (hidden_size[i] // output_size)
                    kernel_mask[i][j][(j % (hidden_size[i] // output_size)) // child_num
                                      + (hidden_size[i + 1] // output_size) * _class] = 1
    else:
        # i層目、j行、k列のノードと親ノードの結合
        for i in range(len(kernel_mask)):
            for j in range(hidden_size[i]):
                if i == 0:
                    image_size = int(np.sqrt(hidden_size[i]))
                    for _class in range(output_size):
                        """
                        print("i:{} j:{} _class:{} -> parent_pos:{} + {} parent:{}"
                              .format(i, j, _class,
                                      (j // (child_num * image_size)) * math.ceil(image_size / child_num),
                                      (j % image_size) // child_num,
                                      (hidden_size[i + 1] // output_size) * _class))
                        """
                        kernel_mask[i][j][(j // (child_num * image_size)) * math.ceil(image_size / child_num)
                                          + (j % image_size) // child_num
                                          + (hidden_size[i + 1] // output_size) * _class] = 1
                else:
                    image_size = int(np.sqrt(hidden_size[i] // output_size))
                    _class = j // (hidden_size[i] // output_size)
                    """
                    print("i:{} j:{} _class:{} -> parent_pos:{} + {} parent:{} image_size:{}"
                          .format(i, j % (hidden_size[i] // output_size), _class,
                                  (j % (hidden_size[i] // output_size) // (child_num * image_size)) * math.ceil(image_size / child_num),
                                  (j % (hidden_size[i] // output_size) % image_size) // child_num,
                                  (hidden_size[i + 1] // output_size) * _class,
                                  image_size))
                    """
                    kernel_mask[i][j][
                        ((j % (hidden_size[i] // output_size)) // (child_num * image_size)) * math.ceil(image_size / child_num)
                        + ((j % (hidden_size[i] // output_size)) % image_size) // child_num
                        + (hidden_size[i + 1] // output_size) * _class] = 1
    if show_mask:
        for i in range(len(kernel_mask)):
            print("kernel_mask[{}]:{}".format(i, kernel_mask[i]))
    return kernel_mask


def write_result(path_w, str_list):
    # from datetime import datetime
    # path_w = os.getcwd() + r"\result\{}".format(datetime.now().strftime("%Y%m%d%H%M%S"))
    with open(path_w, mode='w') as f:
        for s in str_list:
            # print(s)
            f.write(str(s) + "\n")
    # with open(path_w) as f:
        # print(f.read())


if __name__ == '__main__':
    write_result()
    exit()
    # mlp = tree_mlp(13, 3)
    print(softmax([11, -20, 22]))
    print(softmax([-19, -20, 22]))
    print(softmax([0.40693525878330566, -20.82664044877567, 22.328847296583998]))

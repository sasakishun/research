#!/usr/bin/env python
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
from keras import backend as K
# import numpy as np
# from keras.utils import np_utils
import argparse
# import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import random
from visualization import visualize, hconcat_resize_min, vconcat_resize_min
from _tree_functions import *
# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
from draw_architecture import *

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)

### モデル量子化
# from keras.models import load_model
# from keras_compressor.compressor import compress
### モデル量子化
# old_session = KTF.get_session()
# session = tf.Session('')
# KTF.set_session(session)
# KTF.set_learning_phase(1)
###

Height, Width = 28, 28
Channel = 1
output_size = 10
input_size = Height * Width * Channel
wSize = 20
dataset = ""
binary_flag = False
tree_flag = False
binary_target = None
pruning_rate = -1
dataset_category = 10
no_mask = False
dense_size = [64, 60, 32, 16, 10]
CHILD_NUM = 2


def krkopt_data():
    _train = [[], []]
    file_data = open("krkopt.data", "r")
    # まず１行読み込む
    line = file_data.readline()
    # while文を使い、読み込んだ行の表示と次の行の取得を行う
    while line:  # lineが取得できる限り繰り返す
        print(line)
        _line = (line.rstrip("\n")).split(",")
        print(_line)
        for i in range(len(_line)):
            if _line[i] == "a":
                _line[i] = 0
            elif _line[i] == "b":
                _line[i] = 1
            elif _line[i] == "c":
                _line[i] = 2
            elif _line[i] == "draw":
                _line[i] = 17
            elif _line[i] == "zero":
                _line[i] = 0
            elif _line[i] == "one":
                _line[i] = 1
            elif _line[i] == "two":
                _line[i] = 2
            elif _line[i] == "three":
                _line[i] = 3
            elif _line[i] == "four":
                _line[i] = 4
            elif _line[i] == "five":
                _line[i] = 5
            elif _line[i] == "six":
                _line[i] = 6
            elif _line[i] == "seven":
                _line[i] = 7
            elif _line[i] == "eight":
                _line[i] = 8
            elif _line[i] == "nine":
                _line[i] = 9
            elif _line[i] == "ten":
                _line[i] = 10
            elif _line[i] == "eleven":
                _line[i] = 11
            elif _line[i] == "twelve":
                _line[i] = 12
            elif _line[i] == "thrteen":
                _line[i] = 13
            elif _line[i] == "fourteen":
                _line[i] = 14
            elif _line[i] == "fifteen":
                _line[i] = 15
            elif _line[i] == "sixteen":
                _line[i] = 16
            else:
                _line[i] = int(_line[i])
            print(_line[:6])
            _train[0].append(_line[:6])
            _train[1].append(_line[6])
        line = file_data.readline()
    # 開いたファイルを閉じる
    file_data.close()
    X_train, X_test, y_train, y_test = \
        train_test_split(_train[0], _train[1], test_size=0.2, train_size=0.8, shuffle=True, random_state=1)
    y_train = np_utils.to_categorical(y_train, 17)
    y_test = np_utils.to_categorical(y_test, 17)

    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def iris_data():
    X_train, X_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.2, train_size=0.8, shuffle=True, random_state=1)
    X_train, y_train = augumentaton(X_train, y_train)
    X_test, y_test = augumentaton(X_test, y_test)
    y_train = np_utils.to_categorical(y_train, 3)
    y_test = np_utils.to_categorical(y_test, 3)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def normalize(X_train, X_test):
    # 正規化
    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    X_train = mmsc.fit_transform(X_train)
    X_test = mmsc.transform(X_test)
    return X_train, X_test


def standardize(X_train, X_test):
    # 標準化
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train = stdsc.fit_transform(X_train)
    X_test = stdsc.transform(X_test)
    return X_train, X_test


def wine_data(train_flag=True):
    ### .dataファイルを読み込む
    _data = open(os.getcwd() + r"\wine.data", "r")
    lines = _data.readlines()
    ### .dataファイルを読み込む

    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む
    _train = []
    _target = []
    for line in lines:
        line = stringToList(line, ",")
        _target.append(int(line[0]) - 1)  # 今回はリストの先頭がクラスラベル(1 or 2 or 3)
        _train.append([float(i) for i in line[1:]])  # それ以外は訓練データ
    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む

    # _train, _target = augumentaton(_train, _target)
    X_train, X_test, y_train, y_test = \
        train_test_split(np.array(_train), np.array(_target), test_size=0.1, train_size=0.9, shuffle=True,
                         random_state=1)
    ### 訓練時は全クラスでデータ数そろえる -> BatchNormalixationするときには外すべき？
    # if train_flag:
    # X_train, y_train = augumentaton(list(X_train), y_train)
    # X_test, y_test = augumentaton(list(X_test), y_test)
    ### 全クラスでデータ数そろえる

    ### 各列で正規化
    # X_train, X_test = normalize(X_train, X_test)
    X_train, X_test = standardize(X_train, X_test)
    ### 各列で正規化

    y_train = np_utils.to_categorical(y_train, 3)
    y_test = np_utils.to_categorical(y_test, 3)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    # print("X_train:\n{} \nX_test:\n{}".format(X_train, X_test))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    print("-> X_max in train :{}".format(np.amax(X_train, axis=0)))
    print("-> X_max in test  : {}".format(np.amax(X_test, axis=0)))
    # print("X_train:\n{}".format(X_train))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def augumentaton(train, target):
    # train : リスト
    # target : スカラー値
    samples = [[[], []] for _ in range(dataset_category)]
    for i in range(len(train)):
        samples[target[i]][0].append(train[i])
        samples[target[i]][1].append(target[i])
    _max = max([len(samples[i][0]) for i in range(len(samples))])
    print("samples_num: {}".format([len(samples[i][0]) for i in range(len(samples))]))
    for i in range(len(samples)):
        if len(samples[i][0]) < _max:
            scale = _max // len(samples[i][0]) - 1
            origin = copy.deepcopy(samples[i])
            # print("i:{} scale:{}".format(i, scale))
            for j in range(scale):
                # print(samples[i][0])
                samples[i][0] += origin[0]
                samples[i][1] += origin[1]
            samples[i][0] += origin[0][:_max - len(samples[i][0])]
            samples[i][1] += origin[1][:_max - len(samples[i][1])]
    print("     augumentated -> {}".format([len(samples[i][0]) for i in range(len(samples))]))
    train = []
    target = []
    for i in range(len(samples)):
        for j in range(len(samples[i][0])):
            train.append(samples[i][0][j])
            target.append(samples[i][1][j])
    return np.array(train), np.array(target)


def balance_data():
    ### .dataファイルを読み込む
    _data = open(r"C:\Users\papap\Documents\research\DCGAN_keras-master\balance-scale.data", "r")
    lines = _data.readlines()
    ### .dataファイルを読み込む

    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む
    _train = []
    _target = []
    for line in lines:
        line = stringToList(line, ",")
        # 今回はリストの先頭がクラスラベル(L or B or R)
        if line[0] == "L":
            _target.append(0)
        elif line[0] == "B":
            _target.append(1)
        elif line[0] == "R":
            _target.append(2)
        else:
            print("Dataset Error")
            exit()
        _train.append([float(i) for i in line[1:]])  # それ以外は訓練データ
    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む

    normalize_array = np.full((len(_train), 4), 1 / 5)
    _train *= normalize_array
    X_train, X_test, y_train, y_test = \
        train_test_split(np.array(_train), np.array(_target), test_size=0.2, train_size=0.8, shuffle=True,
                         random_state=1)
    X_train, y_train = augumentaton(X_train, y_train)
    usable = [0, 1, 2]
    if binary_flag:
        X_train, X_test, y_train, y_test = digits_data_binary(usable, X_train, X_test, y_train, y_test)
    else:
        y_train = np_utils.to_categorical(y_train, 3)
        y_test = np_utils.to_categorical(y_test, 3)
    ### 各列で正規化
    # X_train, X_test = normalize(X_train, X_test)
    # X_train, X_test = standardize(X_train, X_test)
    ### 各列で正規化

    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("X_train:\n{} \nX_test:\n{}".format(X_train, X_test))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    print("-> X_max in train :{}".format(np.amax(X_train, axis=0)))
    print("-> X_max in test  : {}".format(np.amax(X_test, axis=0)))
    # print("X_train:\n{}".format(X_train))
    # exit()
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def digits_data(binary_flag=False, train_flag=True):
    # if binary_target is None:
    # print("Error : Please input binary_target")
    # exit()
    usable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # random.sample([int(i) for i in range(10)], 2)
    # X_train, X_test, y_train, y_test = \
    # train_test_split(digits.data, digits.target, test_size=0.2, train_size=0.8, shuffle=True, random_state=1)
    _X_train, _y_train = digits.data, digits.target
    X_train, y_train = [], []
    for data, target in zip(_X_train, _y_train):
        for _usable in usable:
            if target == _usable:
                X_train.append(data)
                y_train.append(target)
                break
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.2, train_size=0.8, shuffle=True, random_state=2)
    ### データ数偏り補正
    if tree_flag:
        X_train, y_train = augumentaton(X_train, y_train)
        X_test, y_test = augumentaton(X_test, y_test)
    ### データ数偏り補正

    X_train, X_test = normalize(X_train, X_test)
    print("digit.data:{}".format(digits.data.shape))
    print("digit.target:{}".format(digits.target.shape))
    print("X_train:{}".format(X_train.shape))
    print("y_train:{}".format(y_train.shape))
    if binary_flag:
        X_train, X_test, y_train, y_test = digits_data_binary(usable, X_train, X_test, y_train, y_test)
    else:
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def digits_data_binary(usable, _X_train, _X_test, _y_train, _y_test):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(_y_train)):
        if _y_train[i] == usable[binary_target]:
            # _y_train[i] = 1
            for j in range(len(usable) - 1):  # データ数をそろえる処理
                y_train.append([1])
                X_train.append(_X_train[i])
        else:
            # _y_train[i] = 0
            y_train.append([0])
            X_train.append(_X_train[i])
    for i in range(len(_y_test)):
        if _y_test[i] == usable[binary_target]:
            # _y_test[i] = 1
            for j in range(len(usable) - 1):
                y_test.append([1])
                X_test.append(_X_test[i])
        else:
            # _y_test[i] = 0
            y_test.append([0])
            X_test.append(_X_test[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    return X_train, X_test, y_train, y_test


def parity_data():
    _X_train, _y_train = [[int(1 if random.random() > 0.5 else 0) for _ in range(input_size)]
                          for _ in range(2000)], [0 for _ in range(2000)]
    # _X_train, _y_train = [[int(1 if random.random() > 0.5 else 0) if i == 0 else 0 for i in range(input_size)]
    # for _ in range(1000)], [0 for _ in range(1000)]
    for i in range(len(_y_train)):
        if sum(_X_train[i]) % 2 == 1:
            _y_train[i] = 1
            # for i in range(len(_X_train)):
            # print("_X_train:{} _y_train:{}".format(sum(_X_train[i]), _y_train[i]))
    X_train, y_train = _X_train, _y_train
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.2, train_size=0.8, shuffle=True, random_state=2)
    ### データ数偏り補正
    X_train, y_train = augumentaton(X_train, y_train)
    X_test, y_test = augumentaton(X_test, y_test)
    ### データ数偏り補正

    X_train, X_test = normalize(X_train, X_test)
    print("X_train:{}".format(X_train.shape))
    print("y_train:{}".format(y_train.shape))
    # X_train = np.array([[[__data] for __data in _data] for _data in X_train])
    # X_test = np.array([[[__data] for __data in _data] for _data in X_test])
    y_train = np_utils.to_categorical(y_train, dataset_category)
    y_test = np_utils.to_categorical(y_test, dataset_category)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 28 * 28))
    X_test = X_test.reshape((X_test.shape[0], 28 * 28))
    # クラス分類モデル用に追加
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite


def cifar10_data():
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 32 * 32 * 3))
    X_test = X_test.reshape((X_test.shape[0], 32 * 32 * 3))
    # クラス分類モデル用に追加
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite

def my_tqdm(ite):
    ### 学習進行状況表示
    con = '|'
    _div = cf.Save_train_step // 20
    if ite % cf.Save_train_step != 0:
        for i in range((ite % cf.Save_train_step) // _div):
            con += '>'
        for i in range(cf.Save_train_step // _div - (ite % cf.Save_train_step) // _div):
            con += ' '
    else:
        for i in range(cf.Save_train_step // _div):
            con += '>'
    con += '| '
    ### 学習進行状況表示
    return con


def getdata(dataset, binary_flag, train_frag=True):
    if dataset == "iris":
        return iris_data()
    elif dataset == "mnist":
        return mnist_data()
    elif dataset == "digits":
        return digits_data(binary_flag=binary_flag, train_flag=train_frag)
    elif dataset == "wine":
        return wine_data(train_flag=train_frag)
    elif dataset == "balance":
        return balance_data()
    elif dataset == "parity":
        return parity_data()
    elif dataset == "cifar10":
        return cifar10_data()


def mask(masks, batch_size):
    ### 1バッチ分のmask生成
    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    # print("masks({}):{}".format(type(masks), [np.shape(i) for i in masks]))
    # print("_mask({}):{}".format(type(_mask), [np.shape(i) for i in _mask]))
    if type(masks) is list:
        _mask = masks
    else:
        _mask[1] = np.array(masks)
    ### 1バッチ分のmask生成

    ###各maskをミニバッチサイズでそれぞれ複製
    return [np.array([_mask[i] for _ in range(batch_size)]) for i in range(len(_mask))]
    # mask[0]*32, mask[1]*32,...を返す
    ###各maskをミニバッチサイズでそれぞれ複製


def show_weight(weights):
    print("weights:{}".format([np.shape(i) for i in weights]))


def load_concate_masks(active_true=False):
    ### 保存されているmaskにおける、and集合を返す
    g_mask = np.zeros(dense_size[1])
    for i in range(dataset_category):
        _g_mask = np.load(cf.Save_layer_mask_path[i])
        if g_mask.shape != _g_mask.shape:
            np.save(cf.Save_layer_mask_path[i], g_mask)
            _g_mask = g_mask
        # print("g_mask:{}\n{}".format(g_mask.shape, g_mask))
        # print("_g_mask:{}\n{}".format(_g_mask.shape, _g_mask))
        for j in range(g_mask.shape[0]):
            if g_mask[j] == 1 or _g_mask[j] == 1:
                g_mask[j] = 1
    if not active_true:
        for i in range(len(g_mask)):
            g_mask[i] = (g_mask[i] + 1) % 2
    if no_mask:
        return np.ones(dense_size[1])
    else:
        return g_mask


def generate_syncro_weights(binary_classify, size_only=False):
    """
    syncro_weights = [np.zeros((input_size, wSize)), np.zeros((wSize, ))]
    for i in range(dataset_category):
        binary_classify.load_weights(cf.Save_binary_classify_path[:-3] + str(i) + cf.Save_binary_classify_path[-3:])
        syncro_weights[0] = np.add(syncro_weights[0], binary_classify.get_weights()[0])
        syncro_weights[1] = np.add(syncro_weights[1], (binary_classify.get_weights()[1])*np.load(cf.Save_layer_mask_path[i]))
        # syncro_weights[1] = np.add(syncro_weights[1], binary_classify.get_weights()[1])
        print("bias\n{}\n".format(binary_classify.get_weights()[1]))
    """
    # if size_only:
    # return [np.ones((wSize, input_size)), np.ones(wSize)]
    syncro_weights = [[], []]
    active_nodes_num = [0 for _ in range(output_size)]
    for i in range(dataset_category):
        try:
            binary_classify.load_weights(cf.Save_binary_classify_path[:-3] + str(i) + cf.Save_binary_classify_path[-3:])
        except:
            return [np.ones((dense_size[1], input_size)), np.ones(dense_size[1])]
        _mask = np.load(cf.Save_layer_mask_path[i])
        print("mask[{}]:{}".format(i, _mask))
        for j in range(len(_mask)):
            if _mask[j] == 1:
                active_nodes_num[i] += 1
                syncro_weights[0].append((binary_classify.get_weights()[0][:, j]).T)
                syncro_weights[1].append(binary_classify.get_weights()[1][j])
        if i == dataset_category - 1:
            show_weight(binary_classify.get_weights())
    syncro_weights = [np.array(syncro_weights[0]).T, np.array(syncro_weights[1])]
    print("\n\n\n\nsyncro_weights:{}\n{}".format(syncro_weights[0].shape, syncro_weights[0]))
    print("syncro_bias   :{}\n{}".format(syncro_weights[1].shape, syncro_weights[1]))
    return syncro_weights, active_nodes_num


def inputs_z(X_test, g_mask_1):
    return list(np.array([X_test])) + mask(g_mask_1, len(X_test))


def binarize_inputs_z(X_train, X_test):  # 入力データを変形[i]*64->[i,j]*32
    def binarize(data):
        return [[[data[i][2 * j], data[i][2 * j + 1]] for j in range(len(data[i]) // 2)] for i in range(len(data))]

    return binarize(X_train), binarize(X_test)


def separate_inputs_z(data):
    # return data
    return [np.array([[data[j][i]] for j in range(np.shape(data)[0])]) for i in range(np.shape(data)[1])]


# 入力 : np.array 2つ 出力: シャッフルされたnp.array 2つ
def shuffle_data(X_data, y_data):
    p = np.random.permutation(len(X_data))
    return X_data[p], y_data[p]


# mlpのweightにmaskをかける
def multiple_mask_to_model(_mlp, kernel_mask=None, bias_mask=None):
    _weight = get_kernel_and_bias(_mlp)  # _mlp.get_weights()
    if kernel_mask is not None:
        for i in range(len(_weight) // 2):
            print("multiple num :{}".format(i))
            _weight[i * 2] *= kernel_mask[i]
    if bias_mask is not None:
        for i in range(len(_weight) // 2):
            print("multiple num :{}".format(i))
            _weight[i * 2 + 1] *= bias_mask[i]
    # BNを考慮して重みセット
    _mlp = set_weights(_mlp, _weight)  # _mlp.set_weights(_weight)
    return _mlp


# _weightsの重みリストの二次元配列の出現間隔が3以上ならBN使用と判別
def batchNormalization_is_used(_weights):
    kernel_start = None
    kernel_span = None
    for i in range(len(_weights)):
        if _weights[i].ndim == 2:
            if kernel_start is None:
                kernel_start = i
            else:
                kernel_span = i - kernel_start
                break
    if kernel_span != 2:
        return True
    else:
        return False


# BN使用可において,kernel始まりのインデックスと,[BN,kernel,bias]の1層あたりの重み長さを返す
def get_kernel_start_index_and_set_size(_mlp):
    kernel_start = 0
    set_size = 6
    _weights = _mlp.get_weights()
    for i, _weight in enumerate(_weights):
        if _weight.ndim == 2:
            kernel_start = i
            break
    return kernel_start, set_size


# mlpのmaskを再計算・更新
def update_mask_of_model(_mlp):
    kernel_mask, bias_mask = get_kernel_bias_mask(_mlp)  # mask取得
    _weight = _mlp.get_weights()
    _mlp = myMLP(get_layer_size_from_weight(_weight), kernel_mask=kernel_mask,
                 bias_mask=bias_mask)  # mask付きモデル宣言
    _mlp.set_weights(_weight)  # 学習済みモデルの重みセット
    return _mlp


# 入力重み_weightsからNN構造(各層のノード数)を返す
def get_layer_size_from_weight(_weights=None):
    if _weights is None:
        d = np.load(cf.Save_np_mlp_path)
        print("np.load(cf.Save_np_mlp_path):{}".format(d))
        return get_layer_size_from_weight(np.load(cf.Save_np_mlp_path))
    else:
        # print("_weights:{}".format(_weights))
        return [np.shape(_weights[0])[0]] + [np.shape(i)[1] for i in _weights if i.ndim == 2]


# プルーニングしmaskも更新
def prune_and_update_mask(_mlp, X_data, y_data):
    _mlp = _weight_pruning(_mlp, X_data, y_data)  # pruning重み取得
    _mlp = update_mask_of_model(_mlp)
    return _mlp


# 各層のノードを見やすいように並び替え&NN構造画像保存
def sort_all_layer(_mlp, X_data=None, y_data=None):
    sorted_weights = _mlp.get_weights()
    for i in range(len(sorted_weights) // 2 - 1):
        sorted_weights = sort_weights(sorted_weights, target_layer=i)
        _mlp.set_weights(sorted_weights)
        if X_data is not None and y_data is not None:
            visualize_network(_mlp.get_weights(),
                              _mlp.evaluate(X_data, y_data)[1],
                              comment="sorted layer:{}".format(i))
    return _mlp


# mlpモデル or weightsリストから、kernelとバイアス(BNパラメータ抜き)を返す
def get_kernel_and_bias(_mlp):
    _weights = model2weights(_mlp)
    # print("\n\n\nget_kernel_and_bias")
    # show_weight(_weights)
    if batchNormalization_is_used(_weights):
        kernel_start, set_size = get_kernel_start_index_and_set_size(_mlp)
        # print("kernel_start:{} set_size:{}".format(kernel_start, set_size))
        kernel_and_bias = []
        for i in range(len(_weights) // set_size):
            kernel_and_bias.append(_weights[i * set_size + kernel_start])
            kernel_and_bias.append(_weights[i * set_size + kernel_start + 1])
        return kernel_and_bias
    else:
        return _weights


# _mlpモデルに重みをセット(BNパラメータ有り無し両対応)
def set_weights(_mlp, _weights):
    kernel_and_bias = _mlp.get_weights()
    kernel_start, set_size = get_kernel_start_index_and_set_size(_mlp)
    # set元でBN使用
    if batchNormalization_is_used(_weights):
        # print("set元でBN使用")
        if batchNormalization_is_used(_mlp.get_weights):
            # print("set先でBN使用")
            _mlp.set_weights(_weights)
        else:
            # print("set先でBNなし")
            for i in range(len(_weights) // set_size):
                kernel_and_bias[2 * i] = _weights[i * set_size + kernel_start]
                kernel_and_bias[2 * i + 1] = _weights[i * set_size + kernel_start + 1]
            _mlp.set_weigths(kernel_and_bias)
    else:
        # print("set元でBNなし")
        if batchNormalization_is_used(_mlp.get_weights()):
            # print("set先でBN使用")
            for i in range(len(_weights) // 2):
                # print("kernel_and_bias[{}]:{} = _weights[{}]:{}".format(
                # i * set_size + kernel_span, kernel_and_bias[i * set_size + kernel_span], 2*i, _weights[2 * i]))
                # print("kernel_and_bias[{}]:{} = _weights[{}]:{}".format(
                # i * set_size + kernel_span + 1, kernel_and_bias[i * set_size + kernel_span + 1], 2*i+1, _weights[2 * i + 1]))
                kernel_and_bias[i * set_size + kernel_start] = _weights[2 * i]
                kernel_and_bias[i * set_size + kernel_start + 1] = _weights[2 * i + 1]
            _mlp.set_weights(kernel_and_bias)
        else:
            # print("set先でBNなし")
            _mlp.set_weights(get_kernel_and_bias(_weights))
    return _mlp


# 入力:mlpオブジェクト->重みを返す,入力:weightsリスト->そのまま返す
def model2weights(_mlp):
    if str(type(_mlp)) == "<class 'keras.engine.training.Model'>":
        return _mlp.get_weights()
    elif str(type(_mlp)) != "list":
        return _mlp
    else:
        print("Error in model2weight : input_type must be Model or weight_list")
        exit()


# maskをキープしたままmodelを学習
def keep_mask_and_fit(model, X_train, y_train, batch_size=32, kernel_mask=None, bias_mask=None, epochs=100):
    weights = model.get_weights()
    # maskを一時退避
    _kernel_mask, _bias_mask = get_kernel_bias_mask(model)
    if kernel_mask is None:
        kernel_mask = _kernel_mask
    if bias_mask is None:
        bias_mask = _bias_mask
    # mask付きモデル定義
    model = myMLP(get_layer_size_from_weight(weights), kernel_mask=kernel_mask,
                  bias_mask=bias_mask, set_weights=weights)
    # コールバック設定
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    # tb_cb = keras.callbacks.TensorBoard(log_dir=".\log", histogram_freq=1) # 謎エラーが発生するため不使用
    # 学習
    valid_num = len(X_train) // 10
    model.fit(X_train[:-valid_num], y_train[:-valid_num], batch_size=batch_size, epochs=epochs,
              validation_data=(X_train[-valid_num:], y_train[-valid_num:]), callbacks=[es_cb])  # 学習
    # maskをmodelの重みに掛け合わせる
    multiple_mask_to_model(model, kernel_mask=kernel_mask, bias_mask=bias_mask)
    return model


# クラスごとに精度を出す
def evaluate_each_class(model, X_train, y_train, X_test, y_test):
    # クラスごと性能評価
    data, target = divide_data(X_train, y_train, dataset_category)
    for _class in range(dataset_category):
        _predict = model.predict(data[_class])
        print("\nclass[{}] acc:{}\n{}".format(
            _class, model.evaluate(data[_class], target[_class]), [np.argmax(k) for k in _predict]))
    total_data = list(data[0])
    for j in range(1, dataset_category):
        total_data += list(data[j])
    print("\ntotal acc:{}\npredict:{}".
          format(model.evaluate(X_train, y_train, batch_size=1),
                 [np.argmax(j) for j in model.predict(np.array(total_data))]))
    print("\ntotal acc_test:{}".
          format(model.evaluate(X_test, y_test, batch_size=1)))
    return


class Main_train():
    def __init__(self):
        pass

    def train(self, load_model=False, use_mbd=False):
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
            = getdata(dataset, binary_flag=binary_flag, train_frag=True)
        # X_train, y_train = shuffle_data(X_train, y_train)

        ### これだとBNが正しく機能するが、内部処理不明 -> 要分析
        K.set_learning_phase(0)
        ### これだとBNが正しく機能する
        kernel_mask = get_tree_kernel_mask(calculate_tree_shape(input_size, output_size, child_num=CHILD_NUM),
                                           child_num=CHILD_NUM, show_mask=True)
        # visualize_network(weights=kernel_mask)

        _mlp = tree_mlp(input_size, dataset_category, kernel_mask=kernel_mask,
                        child_num=CHILD_NUM)  # myMLP(13, [5, 4, 2], 3)
        for i in range(5):
            X_train, y_train = shuffle_data(X_train, y_train)
            # モデル学習
            _mlp = keep_mask_and_fit(_mlp, X_train, y_train, batch_size=cf.Minibatch,
                                     kernel_mask=kernel_mask, bias_mask=None, epochs=cf.Iteration)

            # ネットワーク可視化
            visualize_network(
                weights=get_kernel_and_bias(_mlp),
                comment="just before pruning_stage:{}\n".format(i)
                        + "train:{:.4f} test:{:.4f}".format(_mlp.evaluate(X_train, y_train)[1],
                                                            _mlp.evaluate(X_test, y_test)[1]))
            # magnitude-basedプルーニング
            _mlp = prune_and_update_mask(_mlp, X_train, y_train)
            # プルーニング後の再学習
            _mlp = keep_mask_and_fit(_mlp, X_train, y_train, batch_size=cf.Minibatch,
                                     kernel_mask=kernel_mask, bias_mask=None, epochs=cf.Iteration)

            # クラスごと精度検証
            evaluate_each_class(_mlp, X_train, y_train, X_test, y_test)

            # ネットワーク可視化(プルーニング後)
            visualize_network(
                weights=get_kernel_and_bias(_mlp),
                comment="pruning_stage:{}\n".format(i)
                        + "train:{:.4f} test:{:.4f}".format(_mlp.evaluate(X_train, y_train)[1],
                                                            _mlp.evaluate(X_test, y_test)[1]))

            # モデル保存
            _mlp.save_weights(cf.Save_mlp_path)
            np.save(cf.Save_np_mlp_path, _mlp.get_weights())
            _mlp.load_weights(cf.Save_mlp_path)
            print("train_acc:{}".format(_mlp.evaluate(X_train, y_train)))
            print("test_acc:{}".format(_mlp.evaluate(X_test, y_test)))
        print("_mlp:{}".format([np.shape(i) for i in _mlp.get_weights()]))
        exit()
        """
        hidden_size = get_layer_size_from_weight(_mlp.get_weights())
        from _model_weightGAN import masked_mlp
        masked_mlp_model = masked_mlp(hidden_size[0], hidden_size[1:-1], hidden_size[-1])
        masked_mlp_model.set_weights(get_kernel_and_bias(_mlp))
        for target_layer in range(1, len(masked_mlp_model.get_weights()) // 2):
            print("shrink {}th layer".format(target_layer))
            masked_mlp_model = shrink_mlp_nodes(masked_mlp_model, target_layer,
                                                 X_train, y_train, X_test, y_test,
                                                 only_active_list=False)
        # masked_mlpとshrinkした箇所、kernel_maskが違うため性能が変化する->要実装9/23～
        kernel_mask, bias_mask = get_kernel_bias_mask(masked_mlp_model.get_weights())
        _mlp = myMLP(get_layer_size_from_weight(masked_mlp_model.get_weights()),
                     kernel_mask=kernel_mask, bias_mask=bias_mask)
        set_weights(_mlp, masked_mlp_model.get_weights())
        _mlp = prune_and_update_mask(_mlp, X_train, y_train)
        _mlp.fit(X_train, y_train, batch_size=cf.Minibatch, epochs=100000)  # 学習
        _mlp = update_mask_of_model(_mlp)
        visualize_network(_mlp.get_weights(),
                          _mlp.evaluate(X_test, y_test)[1],
                          comment="unsorted")
        # 全ノードをソート
        # sort_all_layer(_mlp, X_train, y_train)

        ## Save trained model
        _mlp.save_weights(cf.Save_mlp_path)
        np.save(cf.Save_np_mlp_path, _mlp.get_weights())
        return
        """


def show_result(input, onehot_labels, layer1_out, ite, classify, testflag=False, showflag=False, comment=""):
    print("\n{}".format(" test" if testflag else "train"))
    labels_scalar = np.argmax(onehot_labels, axis=1)
    # print("labels_scalar:{}".format(labels_scalar))
    # print("classify:{}".format(classify))
    layer1_outs = [[[], [], []] for _ in range(output_size)]
    for i in range(len(labels_scalar)):
        layer1_outs[labels_scalar[i]][0].append(input[i])  # 入力
        layer1_outs[labels_scalar[i]][1].append(layer1_out[i])  # 中間層出力
        layer1_outs[labels_scalar[i]][2].append(classify[i])  # 分類層出力
    # print("layer1_outs:{}".format(layer1_outs))
    """
    for i in range(len(layer1_outs)):
        # print("\nlabel:{}".format(i))
        for j in range(len(layer1_outs[i][0])):
            # print("{}".format(np.argmax(layer1_outs[i][2][j])))
            if dataset == "iris":
                print("{} -> {} -> {} :{}".format(np.array(layer1_outs[i][0][j]),
                                                  np.array(layer1_outs[i][1][j]),
                                                  np.array(layer1_outs[i][2][j]),
                                                  "" if np.argmax(layer1_outs[i][2][j]) == i else "x"))
            else:
                print("-> {} -> {} :{}".format(np.array(layer1_outs[i][1][j]),
                                               np.array(layer1_outs[i][2][j]),
                                               "" if np.argmax(layer1_outs[i][2][j]) == i else "x"))
                # print("\nlabel:{} input / {}\n{}".format(i, len(layer1_outs[i][0]), np.array(layer1_outs[i][0])))
                # print("label:{} layer1_outs / {}\n{}".format(i, len(layer1_outs[i][1]), np.array(layer1_outs[i][1])))
                # print("label:{} x_outs / {}\n{}".format(i, len(layer1_outs[i][2]), np.array(layer1_outs[i][2])))
    """
    return visualize([_outs[1] for _outs in layer1_outs], [_outs[2] for _outs in layer1_outs], labels_scalar, ite,
                     testflag, showflag=showflag, comment=comment)
    # visualize([_outs[1] for _outs in layer1_outs], [_outs[2] for _outs in layer1_outs], labels_scalar, ite,
    # testflag, showflag=False)


def weight_pruning(_weights, test_val_loss, binary_classify, X_test, g_mask_1, y_test, freezed_classify_1, classify,
                   hidden_layers, pruned_test_val_loss):
    ### 重みプルーニング
    global pruning_rate
    print("pruning_rate:{}".format(pruning_rate))
    if pruning_rate >= 0:
        while (pruned_test_val_loss[1] > test_val_loss[1] * 0.95) and pruning_rate < 10:
            ### 精度98%以上となる重みを_weights[0]に確保
            _weights[0] = copy.deepcopy(_weights[1])
            ### 精度98%以上となる重みを_weights[0]に確保

            ### プルーニング率を微上昇させ性能検証
            pruning_rate += 0.01
            non_zero_num = 0
            pruning_layers = np.shape(_weights[1])[0]
            if binary_flag:
                pruning_layers = 2
            for i in range(pruning_layers):
                if (not binary_flag) and i < 2:
                    # 10クラス分類時には第一中間層重みはプルーニングしない
                    continue
                if _weights[1][i].ndim == 2:  # 重みプルーニング
                    print("np.shape(_weights[1][{}]):{}".format(i, np.shape(_weights[1][i])))
                    for j in range(np.shape(_weights[1][i])[0]):
                        for k in range(np.shape(_weights[1][i])[1]):
                            if abs(_weights[1][i][j][k]) < pruning_rate:
                                _weights[1][i][j][k] = 0.
                    non_zero_num += np.count_nonzero(_weights[1][i] > 0)
                    print("weights[{}]:{} (>0)".format(i, np.count_nonzero(_weights[1][i] > 0)))
                    if non_zero_num == 0:
                        break
                else:  # バイアスプルーニング
                    for j in range(np.shape(_weights[1][i])[0]):
                        if abs(_weights[1][i][j]) < pruning_rate:
                            _weights[1][i][j] = 0.
            if binary_flag:
                binary_classify.set_weights(_weights[1])
                pruned_test_val_loss = binary_classify.evaluate(inputs_z(X_test, g_mask_1), y_test)  # [0.026, 1.0]
            else:
                freezed_classify_1.set_weights(_weights[1])
                pruned_test_val_loss = freezed_classify_1.evaluate(inputs_z(X_test, g_mask_1), y_test)
            print("pruning is done")
            ### プルーニング率を微上昇させ性能検証
        # print("cf.Save_binary_classify_path:{}".format(cf.Save_binary_classify_path))
        if binary_flag:
            binary_classify.set_weights(_weights[0])
            binary_classify.save_weights(cf.Save_binary_classify_path)
            classify.save_weights(cf.Save_classify_path)
            for i in range(len(hidden_layers)):
                hidden_layers[i].save_weights(cf.Save_hidden_layers_path[i])
        else:
            freezed_classify_1.set_weights(_weights[0])
            # freezed_classify_1.save(cf.Save_freezed_classify_1_path)
    ### 重みプルーニング
    return _weights, test_val_loss, binary_classify, g_mask_1, freezed_classify_1, classify, hidden_layers, pruned_test_val_loss


def get_active_nodes(binary_classify, X_train, y_train):
    ### 第1中間層ノードプルーニング
    active_nodes = []  # 整数リスト　g_maskはone-hot表現
    acc_list = []
    g_mask_1 = load_concate_masks(False)  # activeでないノード=1
    ### 欠落させると精度が落ちるノードを検出
    pruned_train_val_acc = \
        binary_classify.evaluate(inputs_z(X_train, g_mask_1), y_train)[1]
    for i in range(dense_size[1]):
        # g_mask_1 = np.ones(wSize)
        g_mask_1[i] = 0
        _acc = \
            binary_classify.evaluate(inputs_z(X_train, g_mask_1), y_train)[1]  # [0.026, 1.0]
        acc_list.append([_acc, i])
        if _acc < pruned_train_val_acc * 0.999:
            active_nodes.append(i)
        g_mask_1[i] = 1
    if len(active_nodes) == 0:
        for i in range(dense_size[1]):
            _acc_list = sorted(acc_list)
            if g_mask_1[_acc_list[-i - 1][1]] == 1:
                active_nodes.append(_acc_list[-1][1])
                break
                # g_mask_1[sorted(acc_list)[-1][1]] = 0
    concated_active = np.zeros(dense_size[1])
    for i in active_nodes:
        concated_active[i] = 1
    np.save(cf.Save_layer_mask_path[binary_target], concated_active)  # g_mask_1)
    g_mask_1 = concated_active  # load_concate_masks(active_true=True)
    print("active_nodes:{}".format(active_nodes))
    ### 第1中間層ノードプルーニング
    return g_mask_1  # active_node箇所だけ1


def get_active_node_non_mask(model, X_train, y_train, target_layer):
    active_nodes = [[] for _ in range(output_size)]  # 整数リスト　g_maskはone-hot表現
    acc_list = [[] for _ in range(output_size)]
    target_layer *= 2
    g_mask_1 = np.ones(np.shape(model.get_weights())[target_layer][0])  # load_concate_masks(False)  # activeでないノード=1

    for i in range(len(X_train)):
        ### 欠落させると精度が落ちるノードを検出
        pruned_train_val_acc = model.evaluate([X_train, mask(g_mask_1, len(X_train[i]))], y_train[i])[1]
        for j in range(len(g_mask_1)):
            g_mask_1[i] = 0
            _acc = model.evaluate([X_train[j], mask(g_mask_1, len(X_train[j]))], y_train[j])[1]
            acc_list.append([_acc, i])
            if _acc < pruned_train_val_acc * 0.999:
                active_nodes.append(i)
            g_mask_1[i] = 1
        if len(active_nodes) == 0:
            for i in range(dense_size[1]):
                _acc_list = sorted(acc_list)
                if g_mask_1[_acc_list[-i - 1][1]] == 1:
                    active_nodes.append(_acc_list[-1][1])
                    break
                    # g_mask_1[sorted(acc_list)[-1][1]] = 0
    concated_active = np.zeros(dense_size[1])
    for i in active_nodes:
        concated_active[i] = 1
    np.save(cf.Save_layer_mask_path[binary_target], concated_active)  # g_mask_1)
    g_mask_1 = concated_active  # load_concate_masks(active_true=True)
    print("active_nodes:{}".format(active_nodes))
    return g_mask_1  # active_node箇所だけ1


def divide_data(X_test, y_test, dataset_category):
    _X_test = [[] for _ in range(dataset_category)]
    _y_test = [[] for _ in range(dataset_category)]
    for data, target in zip(X_test, y_test):
        _X_test[np.argmax(target)].append(np.array(data))
        _y_test[np.argmax(target)].append(np.array(target))
    for i in range(dataset_category):
        _X_test[i] = np.array(_X_test[i])
        _y_test[i] = np.array(_y_test[i])
    return _X_test, _y_test


def shrink_nodes(model, target_layer, X_train, y_train, X_test, y_test, only_active_list=False):
    # model: freezed_classify_1のみ対応
    # 入力 : 全クラス分類モデル(model)、対象レイヤー番号(int)、訓練データ(np.array)、訓練ラベル(np.array)
    # 出力 : 不要ノードを削除したモデル(model)
    target_layer *= 2  # weigthsリストが[重み、バイアス....]となっているため
    weights = model.get_weights()
    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    active_nodes = [[] for _ in range(output_size)]
    X_trains, y_trains = divide_data(X_train, y_train, dataset_category)  # クラス別に訓練データを分割
    for i in range(output_size):  # for i in range(クラス数):
        # i クラスで使用するactiveノード検出 -> active_nodes=[[] for _ in range(len(クラス数))]
        pruned_train_val_acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
        for j in range(len(_mask[target_layer // 2])):
            _mask[target_layer // 2][j] = 0
            _acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
            if _acc < pruned_train_val_acc * 0.999:
                active_nodes[i].append(j)  # activeノードの番号を保存 -> active_nodes[i].append(activeノード)
            _mask[target_layer // 2][j] = 1
    print("active_nodes:{}".format(active_nodes))
    if only_active_list:
        return active_nodes
    usable = [True for _ in range(len(_mask[target_layer // 2]))]  # ソートに使用済みのノード番号リスト
    altered_weights = [[], [], []]  # [np.zeros((weights[target_layer]).shape),
    # np.zeros((weights[target_layer - 1]).shape),
    # np.zeros((weights[target_layer + 2]).shape)]
    for i in range(output_size):
        for j in range(len(active_nodes[i])):
            # print("i:{} j:{} usable:{}".format(i, j, usable))
            if usable[active_nodes[i][j]]:
                used_num = sum(1 for x in usable if not x)
                altered_weights[0].append(weights[target_layer - 2][:, active_nodes[i][j]])
                altered_weights[1].append(weights[target_layer - 1][active_nodes[i][j]])
                altered_weights[2].append(weights[target_layer][active_nodes[i][j]])
            usable[active_nodes[i][j]] = False
    for i in range(len(altered_weights)):
        altered_weights[i] = np.array(altered_weights[i])
        if i == 0:
            altered_weights[0] = altered_weights[0].T
    # print("\naltered_weights:{}\n".format([np.shape(i) for i in altered_weights]))
    # print("\ntaregt_weights:{}\n".format([np.shape(i) for i in weights[target_layer-2:target_layer+1]]))
    weights[target_layer - 2] = altered_weights[0][:, :sum(1 for x in usable if not x)]
    weights[target_layer - 1] = altered_weights[1][:sum(1 for x in usable if not x)]
    weights[target_layer] = altered_weights[2][:sum(1 for x in usable if not x)]
    dense_size[target_layer // 2] = sum(1 for x in usable if not x)
    g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1 \
        = weightGAN_Model(input_size=input_size, wSize=dense_size[1], output_size=output_size, use_mbd=use_mbd,
                          dense_size=dense_size)
    freezed_classify_1.set_weights(weights)

    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    im_architecture = mydraw(weights, freezed_classify_1.evaluate(inputs_z(X_test, _mask), y_test)[1],
                             comment="shrinking layer[{}]".format(target_layer // 2))
    im_h_resize = im_architecture
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture\triple"
    if not os.path.exists(path):
        path = r"C:\Users\xeno\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture\triple"
    path += r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + "_{}_.png".format(target_layer // 2))

    cv2.imwrite(path, im_h_resize)
    print("saved concated graph to -> {}".format(path))
    return freezed_classify_1


def set_dense_size_with_tree(input_size, dataset_category):
    global dense_size
    _dense = 1
    dense_size = []
    while _dense < input_size:
        _dense *= 2
        dense_size.append(_dense)
    dense_size.reverse()
    dense_size = [i * dataset_category for i in dense_size]
    return


def tree_inputs2mlp(X_data, input_size, output_size):
    X_data = list(copy.deepcopy(X_data))
    X_data = [list(i) for i in X_data]
    padding = [0 for _ in range(input_size // output_size - len(X_data[0]))]
    _X_data = []
    for data in X_data:
        _data = []
        for j in range(output_size):
            _data += data + padding
        _X_data.append(_data)
    return np.array(_X_data)


# 重みプルーニングし、モデルを返す
def _weight_pruning(model, X_test, y_test, margin_acc=0.95):
    # _weights = [model.get_weights(), model.get_weights()]
    _weights = [get_kernel_and_bias(model), get_kernel_and_bias(model)]
    pruned_test_val_loss = model.evaluate(X_test, y_test)
    test_val_loss = copy.deepcopy(pruned_test_val_loss)
    # global pruning_rate
    pruning_rate = 0.
    while (pruned_test_val_loss[1] > test_val_loss[1] * margin_acc) and pruning_rate < 10:
        ### 精度98%以上となる重みを_weights[0]に確保
        _weights[0] = copy.deepcopy(_weights[1])
        ### 精度98%以上となる重みを_weights[0]に確保

        ### プルーニング率を微上昇させ性能検証
        pruning_rate += 0.01
        non_zero_num = 0
        for i in range(np.shape(_weights[1])[0]):
            if _weights[1][i].ndim == 2:  # 重みプルーニング
                # print("np.shape(_weights[1][{}]):{}".format(i, np.shape(_weights[1][i])))
                for j in range(np.shape(_weights[1][i])[0]):
                    for k in range(np.shape(_weights[1][i])[1]):
                        if abs(_weights[1][i][j][k]) < pruning_rate:
                            _weights[1][i][j][k] = 0.
                non_zero_num += np.count_nonzero(_weights[1][i] > 0)
            """
            else:  # バイアスプルーニング (BNパラメータもpruningしてしまっている)
                for j in range(np.shape(_weights[1][i])[0]):
                    if abs(_weights[1][i][j]) < pruning_rate:
                        _weights[1][i][j] = 0.
            """
            non_zero_num += np.count_nonzero(_weights[1][i] > 0)
            # print("weights[{}]:{} (>0)".format(i, np.count_nonzero(_weights[1][i] > 0)))
            if non_zero_num == 0:
                break
        # model.set_weights(_weights[1])
        model = set_weights(model, _weights[1])
        pruned_test_val_loss = model.evaluate(X_test, y_test)
        print("pruning is done : magnitude < {} discard".format(pruning_rate))
        ### プルーニング率を微上昇させ性能検証
        # model.set_weights(_weights[0])
        model = set_weights(model, _weights[0])
    return model  # _weights[0]


# weightsのうちkernelとbiasを分けて返す
def separate_kernel_and_bias(weights):
    return [weights[i] for i in range(len(weights)) if i % 2 == 0], \
           [weights[i] for i in range(len(weights)) if i % 2 == 1]


# mlpオブジェクトor重みリストからkernel_maskとbiasマスクを返す
def get_kernel_bias_mask(_mlp):
    weights = get_kernel_and_bias(_mlp)
    # 入力 : np.arrayのリスト　ex) [(13, 5), (5,), (5, 4), (4,), (4, 2), (2,), (2, 3), (3,)]
    return separate_kernel_and_bias([np.where(weight != 0, 1, 0) for weight in weights])


def load_weights_and_generate_mlp():
    _mlp = myMLP(get_layer_size_from_weight())
    _mlp.load_weights(cf.Save_mlp_path)
    return _mlp


# 中間層出力を訓練テスト、正誤データごとに可視化
def show_intermidate_layer_with_datas(_mlp, X_train, X_test, y_train, y_test):
    save_fig = True
    correct_data_train, correct_target_train, incorrect_data_train, incorrect_target_train \
        = show_intermidate_output(X_train, y_train, "train", _mlp, save_fig=save_fig)
    correct_data_test, correct_target_test, incorrect_data_test, incorrect_target_test \
        = show_intermidate_output(X_test, y_test, "test", _mlp, save_fig=save_fig)
    show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                    correct_data_test, correct_target_test,
                                    _mlp, name=["CORRECT_train", "CORRECT_test"], save_fig=save_fig)

    show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                    incorrect_data_train, incorrect_target_train,
                                    _mlp, name=["CORRECT_train", "MISS_train"], save_fig=save_fig)

    visualize_miss_neuron_on_network(_mlp, [correct_data_train, correct_target_train],
                                     [incorrect_data_test, incorrect_target_test],
                                     original_data=[X_train, y_train, X_test, y_test],
                                     name=["CORRECT_train", "MISS_test"])
    visualize_miss_neuron_on_network(_mlp, [correct_data_train, correct_target_train],
                                     [correct_data_test, correct_target_test],
                                     original_data=[X_train, y_train, X_test, y_test],
                                     name=["CORRECT_train", "CORRECT_test"])
    visualize_miss_neuron_on_network(_mlp, [correct_data_train, correct_target_train],
                                     [incorrect_data_train, incorrect_target_train],
                                     original_data=[X_train, y_train, X_test, y_test],
                                     name=["CORRECT_train", "MISS_train"])
    return


# 各層の間違いノード番号を、サンプルごとにまとめ返す
# 入力 : shape(層数,ノード数,クラス数)->サンプル番号リスト
# 出力 : shape(クラス数, サンプル数, 層数)->クラスAサンプルBのC層でのミスノード番号のリスト
def get_miss_nodes(out_of_ranges):
    for _layer in range(len(out_of_ranges)):  # 層番号
        for _class in range(len(out_of_ranges[_layer])):  # クラス番号
            print("out_of_ranges[{}][{}]:{}".format(_layer, _class, out_of_ranges[_layer][_class]))

    # 間違いサンプル数を取得
    class_num = len(out_of_ranges[-1][0])
    missed = [[] for _ in range(class_num)]

    # out_of_rangeに含まれる各クラスのサンプル番号を取得
    for _layer in range(len(out_of_ranges)):  # 層番号
        for _class in range(len(out_of_ranges[_layer])):  # クラス番号
            for _node in range(len(out_of_ranges[_layer][_class])):  # ノード番号(0,1,2,3....)
                for _sample in out_of_ranges[_layer][_class][_node]:
                    missed[_class].append(_sample)
                    # for i in range(class_num):
                    # print("out_of_ranges[-1][{}]:{}".format(i, out_of_ranges[-1][i]))
                    # for j in range(class_num):
                    # missed[i] += out_of_ranges[-1][i][j]
    print("missed:{}".format(missed))
    # missed = [sorted(set(_missed), key=_missed.index) if len(i) > 0 else [] for _missed in missed]
    # missed = [list(set(i)) if len(i) > 0 else [] for i in missed]
    missed = [list(set([j["sample"] for j in i])) if len(i) > 0 else [] for i in missed]
    print("missed:{}".format(missed))

    miss_nodes = [[[[] for _ in range(len(out_of_ranges))]  # 層数
                   for j in range(len(missed[i]))]  # クラスiのミスサンプル数
                  for i in range(class_num)]  # クラス数

    # 各クラスごとに、サンプル番号から、対応するmissed[_class]中のインデックスを返す
    sample_num_to_index = [{} for _ in range(class_num)]
    for _class in range(class_num):
        for _sample in range(len(missed[_class])):
            sample_num_to_index[_class][str(missed[_class][_sample])] = _sample

    print("miss_nodes:{}".format(miss_nodes))
    print("len(miss_nodes):{}".format(len(miss_nodes)))
    print("sample_to_num_index:{}".format(sample_num_to_index))
    # shape(クラス数, ミスサンプルインデックス(0,1,2), 層番号)->その層でのミスノード番号のリスト

    # out_of_rangesからmiss_nodesリストを作成
    for _layer in range(len(out_of_ranges)):  # 層番号
        for _class in range(len(out_of_ranges[_layer])):  # クラス番号
            for _node in range(len(out_of_ranges[_layer][_class])):  # ノード番号(0,1,2,3....)
                for _sample in out_of_ranges[_layer][_class][_node]:  # ミスサンプル番号
                    # print("_class:{}  _sample:{} _layer:{} _node:{}".format(_class, _sample, _layer, _node))
                    miss_nodes[_class][sample_num_to_index[_class][str(_sample["sample"])]][_layer]\
                        .append({"node": _node, "color": _sample["color"], "value":_sample["value"] if "value" in _sample else None})
    return miss_nodes, sample_num_to_index


# 入力 : shape(クラス数, サンプル数, 層数)->クラスAサンプルBのC層でのミスノード番号のリスト
# 出力 : shape(クラス数, サンプル数, 層数, ノード数)->色(ミスノードだけクラス色、それ以外は黒)
def get_neuron_color_list_from_out_of_range_nodes(out_of_ranges, layer_sizes):
    neuron_coloers = [[[[[{"color":"black"}] for _ in range(i)] for i in layer_sizes]
                       for _sample in range(len(out_of_ranges[_class]))]
                      for _class in range(len(out_of_ranges))]
    for _class in range(len(out_of_ranges)):
        for _sample in range(len(out_of_ranges[_class])):
            for _layer in range(len(out_of_ranges[_class][_sample])):
                for _neuron in out_of_ranges[_class][_sample][_layer]:
                    # neuron_coloers[_class][_sample][_layer][_neuron["node"]] = colors[_class]
                    neuron_coloers[_class][_sample][_layer][_neuron["node"]].append(_neuron) # _neuron["color"])  # colors[_class]
    return neuron_coloers


# 入力 : _mlp, 正解対象, 間違い対象, 元データ(train_data, train_target, test_data, test_target, 名前リスト)
# ミスニューロンを明示したネットワーク図を描画
def visualize_miss_neuron_on_network(_mlp, correct, incorrect, original_data, name=["correct_train", "miss_test"]):
    """
    out_of_ranges = show_intermidate_train_and_test(correct[0], correct[1],
                                                    incorrect[0], incorrect[1],
                                                    _mlp, name=name, save_fig=False)  # True)
    """
    each_color = show_intermidate_train_and_test(correct[0], correct[1],
                                                 incorrect[0], incorrect[1],
                                                 _mlp, name=name, save_fig=False, get_each_color=True)
    for _class in range(len(each_color)):
        for _layer in range(len(each_color[_class])):
            for _node in range(len(each_color[_class][_layer])):
                print("each_color[{}][{}][{}]:{}".format(_class, _layer, _node, each_color[_class][_layer][_node]))
    """
    print("each_color:{}".format(each_color))
    print("name:{}".format(name))
    print("correct:{} miss:{}".format(len(correct[0]), len(incorrect[0])))
    exit()
    for _class in range(len(each_color)):
        for _layer in range(len(each_color[_class])):
            for _node in range(len(each_color[_class][_layer])):
                print("out_of_ranges[{}][{}][{}]:{}".format(_class, _layer, _node,
                                                                 out_of_ranges[_class][_layer][_node]))
    """

    # miss_nodes: shape(クラス数, サンプル数, 層数)->クラスAサンプルBのC層でのミスノード番号のリスト
    """
    miss_nodes, sample_num_to_index = get_miss_nodes(out_of_ranges)
    print("sample_num_to_index:{}".format(sample_num_to_index))
    for i, _miss_nodes in enumerate(miss_nodes):
        for j, _miss in enumerate(_miss_nodes):
            print("miss_nodes[{}][{}]:{}".format(i, j, _miss))
    neuron_colors = get_neuron_color_list_from_out_of_range_nodes(miss_nodes,
                                                                  get_layer_size_from_weight(_mlp.get_weights()))
    print("neuron_colors:{}".format(neuron_colors))
    """
    miss_nodes, sample_num_to_index = get_miss_nodes(each_color)
    print("sample_num_to_index:{}".format(sample_num_to_index))
    print("miss_nodes:{}".format(miss_nodes))
    neuron_colors = get_neuron_color_list_from_out_of_range_nodes(miss_nodes,
                                                                  get_layer_size_from_weight(_mlp.get_weights()))
    print("\nneuron_colors")
    for i in range(len(neuron_colors)):
        print(neuron_colors[i])

    X_train = original_data[0]
    y_train = original_data[1]
    X_test = original_data[2]
    y_test = original_data[3]

    # ミスニューロンを明示したネットワーク図を描画
    for _class in range(len(neuron_colors)):
        for _sample in range(len(neuron_colors[_class])):
            _sample_num = int([key for key, val in sample_num_to_index[_class].items() if val == _sample][0])
            print("class:{} sample:{}".format(_class, _sample_num))
            visualize_network(
                weights=get_kernel_and_bias(_mlp),
                comment="{} out of {} class:{}_{}\n".format(name[1], name[0], _class, _sample_num)
                        + "train:{:.4f} test:{:.4f}".format(_mlp.evaluate(X_train, y_train)[1],
                                                            _mlp.evaluate(X_test, y_test)[1]),
                neuron_color=neuron_colors[_class][_sample])
            # print("neuron_colors[{}][{}]\n{}".format(_class, _sample, neuron_colors[_class][_sample]))
    return


class Main_test():
    def __init__(self):
        pass

    def test(self, loadflag=True):
        ### これだとBNが正しく機能するが、内部処理不明 -> 要分析
        K.set_learning_phase(0)
        ### これだとBNが正しく機能する

        ###全結合mlpとの比較
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
            = getdata(dataset, binary_flag=binary_flag, train_frag=True)
        _mlp = load_weights_and_generate_mlp()

        print("_mlp:{}".format([np.shape(i) for i in _mlp.get_weights()]))
        print("train_acc:{}".format(_mlp.evaluate(X_train, y_train)))
        print("train_acc 1samples:{}".format(_mlp.evaluate(X_train, y_train, batch_size=1)))
        print("test_acc:{}".format(_mlp.evaluate(X_test, y_test)))
        _mlp = prune_and_update_mask(_mlp, X_train, y_train)
        # 中間層不要ノード削除
        from _tree_functions import _shrink_nodes
        for target_layer in range(1, len(get_layer_size_from_weight(_mlp.get_weights())) - 1):
            X_train, y_train = shuffle_data(X_train, y_train)
            print("shrink {}th layer".format(target_layer))
            _mlp = _shrink_nodes(_mlp, target_layer, X_train, y_train, X_test, y_test)
            kernel_mask, bias_mask = get_kernel_bias_mask(_mlp)
            _mlp = keep_mask_and_fit(_mlp, X_train, y_train, batch_size=cf.Minibatch,
                                     kernel_mask=kernel_mask, bias_mask=bias_mask, epochs=cf.Iteration)
        # 性能評価
        evaluate_each_class(_mlp, X_train, y_train, X_test, y_test)
        """
        data, target = divide_data(X_train, y_train, dataset_category)
        for i in range(dataset_category):
            _predict = _mlp.predict(data[i])
            print("\nclass[{}]:{}".format(i, [np.argmax(j) for j in _predict]))
        total_data = list(data[0])
        for j in range(1, dataset_category):
            total_data += list(data[j])
        print("\ntotal acc:{}\npredict:{}".
              format(_mlp.evaluate(X_train, y_train, batch_size=1),
                     [np.argmax(j) for j in _mlp.predict(np.array(total_data))]))
        print("\ntotal acc_test:{}".
              format(_mlp.evaluate(X_test, y_test, batch_size=1)))
        """
        show_intermidate_layer_with_datas(_mlp, X_train, X_test, y_train, y_test)
        print("finish")
        exit()

        ###全結合mlpとの比較
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
            = digits_data(binary_flag=True, train_flag=True)
        # getdata(dataset, binary_flag=binary_flag, train_frag=True)
        global dataset_category
        global output_size
        dataset_category = 2
        output_size = 2
        kernel_mask = get_tree_kernel_mask(calculate_tree_shape(input_size, output_size))
        _mlp = tree_mlp(input_size, dataset_category, kernel_mask=kernel_mask)  # myMLP(13, [5, 4, 2], 3)
        for i in range(1):
            p = np.random.permutation(len(X_train))
            X_train, y_train = X_train[p], y_train[p]
            print("y_train:{}".format(y_train))
            # for j in range(cf.Iteration):
            # print("ite:{} - {}/{}".format(i, j, cf.Iteration))
            _mlp.fit(X_train, y_train, batch_size=cf.Minibatch, epochs=1000)  # 学習
            _weight = _mlp.get_weights()
            for j in range(len(_weight) // 2):
                _weight[j * 2] *= kernel_mask[j]
            _mlp.set_weights(_weight)
            for j in range(len(_weight)):
                print("weight[{}]\n{}".format(j, _weight[j]))
            visualize_network(
                weights=_mlp.get_weights(),
                comment="just before pruning_stage:{}\n".format(i)
                        + "train:{:.4f} test:{:.4f}".format(_mlp.evaluate(X_train, y_train)[1],
                                                            _mlp.evaluate(X_test, y_test)[1]))
            pruned_weight = _weight_pruning(_mlp, X_train, y_train)  # pruning重み取得
            kernel_mask, bias_mask = get_kernel_bias_mask(pruned_weight)  # mask取得
            _mlp = myMLP(calculate_tree_shape(input_size, output_size), kernel_mask=kernel_mask,
                         bias_mask=bias_mask)  # mask付きモデル宣言
            _mlp.set_weights(pruned_weight)  # 学習済みモデルの重みセット

            visualize_network(
                weights=_mlp.get_weights(),
                comment="pruning_stage:{}\n".format(i)
                        + "train:{:.4f} test:{:.4f}".format(_mlp.evaluate(X_train, y_train)[1],
                                                            _mlp.evaluate(X_test, y_test)[1]))
            for _kernel_mask in kernel_mask:
                print("kernel_mask:{}".format(_kernel_mask))
        print("_mlp:{}".format([np.shape(i) for i in _mlp.get_weights()]))
        hidden_size = calculate_tree_shape(input_size, output_size)
        from _model_weightGAN import masked_mlp
        masked_mlp_model = masked_mlp(hidden_size[0], hidden_size[1:-1], hidden_size[-1])
        masked_mlp_model.set_weights(_mlp.get_weights())
        for target_layer in range(1, len(_mlp.get_weights()) // 2):
            print("shrink {}th layer".format(target_layer))
            masked_mlp_model = shrink_mlp_nodes(masked_mlp_model, target_layer,
                                                X_train, y_train, X_test, y_test,
                                                only_active_list=False)

        # masked_mlpとshrinkした箇所、kernel_maskが違うため性能が変化する->要実装9/23～
        pruned_weight = _weight_pruning(masked_mlp_model.get_weights(), X_train, y_train)  # pruning重み取得
        kernel_mask, bias_mask = get_kernel_bias_mask(pruned_weight)  # mask取得
        _mlp = myMLP([input_size] + [np.shape(i)[0] for i in pruned_weight if i.ndim == 1],
                     kernel_mask=kernel_mask, bias_mask=bias_mask)
        _mlp.set_weights(pruned_weight)
        _mlp.fit(X_train, y_train, batch_size=cf.Minibatch, epochs=1000)  # 学習
        pruned_weight = _weight_pruning(_mlp.get_weights(), X_train, y_train)  # pruning重み取得
        _mlp.set_weights(pruned_weight)
        visualize_network(_mlp.get_weights(),
                          _mlp.evaluate(X_test, y_test)[1],
                          comment="unsorted")
        sorted_weights = _mlp.get_weights()
        for i in range(len(sorted_weights) // 2 - 1):
            sorted_weights = sort_weights(sorted_weights, target_layer=i)
            _mlp.set_weights(sorted_weights)
            visualize_network(_mlp.get_weights(),
                              _mlp.evaluate(X_test, y_test)[1],
                              comment="sorted layer:{}".format(i))
        exit()
        ###

        print("\n\n-----test-----\n\n")
        global dense_size
        ite = 0
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
            = getdata(dataset, binary_flag=binary_flag, train_frag=False)
        if tree_flag:
            tree_model = tree(input_size, dataset_category)
            tree_model.load_weights(cf.Save_tree_path)
            train_val_loss = tree_model.evaluate(separate_inputs_z(X_train), y_train)
            test_val_loss = tree_model.evaluate(separate_inputs_z(X_test), y_test)
            print("train loss:{}".format(train_val_loss))
            print("test  loss:{}".format(test_val_loss))
            set_dense_size_with_tree(input_size, dataset_category)
            mlp_model = mlp(dense_size[0], dense_size[1:], output_size)
            original_X_train = copy.deepcopy(X_train)
            original_X_test = copy.deepcopy(X_test)
            X_train = tree_inputs2mlp(X_train, dense_size[0], output_size)
            X_test = tree_inputs2mlp(X_test, dense_size[0], output_size)
        else:
            g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1 \
                = weightGAN_Model(input_size=input_size, wSize=dense_size[1], output_size=output_size, use_mbd=use_mbd,
                                  dense_size=dense_size)
            g_mask_1 = load_concate_masks(active_true=(not binary_flag))  # np.load(cf.Save_layer_mask_path)
            # g_mask_1 = np.ones(wSize)
            print("g_mask(usable):{}".format(g_mask_1))

        if loadflag:
            if tree_flag:
                tree_model.load_weights(cf.Save_tree_path)
                _tree_weights = []  # tree_model.get_weights()
                for layer in tree_model.layers:
                    if layer.name[:5] == "dense":
                        _tree_weights.append([layer.name, layer.get_weights()])
                _tree_weights.sort()
                tree_weights = []
                for _weights in _tree_weights:
                    for i in range(2):
                        tree_weights.append(_weights[1][i])
                # tree_weights = [np.ones(i.shape) for i in tree_weights]
                mlp_weights = mlp_model.get_weights()
                mlp_weights = convert_weigths_tree2mlp(tree_weights, [np.zeros(_mlp.shape) for _mlp in mlp_weights],
                                                       input_size, output_size, dense_size)
                # visualize_network(mlp_weights, comment="tree architecuture")
                mlp_model.set_weights(mlp_weights)
                show_weight(mlp_model.get_weights())
                test_val_loss = mlp_model.evaluate(X_test, y_test)
                train_val_loss = mlp_model.evaluate(X_train, y_train)
                print("test_loss:{}".format(test_val_loss))
                print("train_loss:{}".format(train_val_loss))
                pruned_test_val_loss = copy.deepcopy(test_val_loss)
                pruned_train_val_loss = copy.deepcopy(train_val_loss)
                _weights = [mlp_model.get_weights(), mlp_model.get_weights()]
                active_nodes_num = -1
                non_active_neurons = None  # non_active_in_tree2mlp()
            elif binary_flag:
                g.load_weights(cf.Save_g_path)
                d.load_weights(cf.Save_d_path)
                c.load_weights(cf.Save_c_path)
                classify.load_weights(cf.Save_classify_path)
                binary_classify.load_weights(cf.Save_binary_classify_path)
                print("load:{}".format(cf.Save_binary_classify_path))
                for i in range(len(hidden_layers)):
                    hidden_layers[i].load_weights(cf.Save_hidden_layers_path[i])
                _weights = [binary_classify.get_weights(), binary_classify.get_weights()]
                test_val_loss = binary_classify.evaluate(inputs_z(X_test, g_mask_1), y_test)  # [0.026, 1.0]
                pruned_test_val_loss = copy.deepcopy(test_val_loss)
            else:
                syncro_weights, active_nodes_num = generate_syncro_weights(binary_classify, size_only=(not loadflag))
                dense_size[1] = len(syncro_weights[1])
                g_mask_1 = np.ones(dense_size[1])
                g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1 \
                    = weightGAN_Model(input_size=input_size, wSize=dense_size[1], output_size=output_size,
                                      use_mbd=use_mbd, dense_size=dense_size)
                freezed_classify_1.load_weights(cf.Save_freezed_classify_1_path)
                show_weight(freezed_classify_1)
                _weights = [freezed_classify_1.get_weights(), freezed_classify_1.get_weights()]
                test_val_loss = freezed_classify_1.evaluate(inputs_z(X_test, g_mask_1), y_test)
                train_val_loss = freezed_classify_1.evaluate(inputs_z(X_train, g_mask_1), y_train)
                pruned_test_val_loss = copy.deepcopy(test_val_loss)
                pruned_train_val_loss = copy.deepcopy(train_val_loss)

                print("pruned_test_val_loss:{}".format(pruned_test_val_loss))
            ### プルーニングなしのネットワーク構造を描画
            # if not binary_flag:
            visualize_network(
                weights=add_original_input(input_size, output_size, mlp_weights) if tree_flag else _weights[0],
                acc=test_val_loss[1],
                comment=("[{} vs other]".format(binary_target) if binary_flag else "")
                        + " pruned <{:.6f}\n".format(0.)
                        + "active_node:{}".format(active_nodes_num if not binary_flag else "-1"),
                non_active_neurons=non_active_neurons)
            ### プルーニングなしのネットワーク構造を描画

            ### magnitude プルーニング
            if tree_flag:
                _weights = _weight_pruning(mlp_model, X_train, y_train)
                mlp_model.set_weights(_weights)
            else:
                _weights, test_val_loss, binary_classify, g_mask_1, freezed_classify_1, classify, hidden_layers, pruned_test_val_loss \
                    = weight_pruning(_weights, test_val_loss, binary_classify, X_test, g_mask_1, y_test,
                                     freezed_classify_1,
                                     classify, hidden_layers, pruned_test_val_loss)
            # = weight_pruning(_weights, test_val_loss, binary_classify, X_train, g_mask_1, y_train, freezed_classify_1, classify, hidden_layers, pruned_test_val_loss)
            ### magnitude プルーニング

            ### 第1中間層ノードプルーニング
            # active_nodes = []
            if binary_flag:
                g_mask_1 = get_active_nodes(binary_classify, X_train, y_train)
                # else:
                # active_nodes = [-1]# g_mask_1
                ### 第1中間層ノードプルーニング

        if (not loadflag) and (pruning_rate >= 0):
            print("\nError : Please load Model to do pruning")
            exit()
        if tree_flag:
            test_val_loss = mlp_model.evaluate(X_test, y_test)
            weights = mlp_model.get_weights()
        elif binary_flag:
            test_val_loss = binary_classify.evaluate(inputs_z(X_test, g_mask_1), y_test)  # [0.026, 1.0]
            train_val_loss = binary_classify.evaluate(inputs_z(X_train, g_mask_1), y_train)
            # binary_classify.load_weights(cf.Save_binary_classify_path)
            weights = binary_classify.get_weights()  # classify.get_weights()
        else:
            test_val_loss = freezed_classify_1.evaluate(inputs_z(X_test, g_mask_1), y_test)
            weights = freezed_classify_1.get_weights()

        for i in range(len(weights)):
            print(np.shape(weights[i]))
        # classify.summary()

        print("\ntest Acc:{}".format(test_val_loss[1]))
        # print("g_mask_1\n{}".format(np.array(np.nonzero(g_mask_1)).tolist()[0]))
        if binary_flag:
            print("g_mask_in_binary\n{}".format(np.array(np.nonzero(load_concate_masks(active_true=True))).tolist()[0]))

        ### magnitudeプルーニング後のネットワーク構造を描画
        global pruning_rate
        print("pruning_rate:{}".format(pruning_rate))
        _comment = "[{} vs other]\n".format(binary_target) if binary_flag else "full classes test\n"
        _comment += " pruned <{:.4f} ".format(pruning_rate)
        _comment += "active_node:{}".format((np.array(np.nonzero(g_mask_1)).tolist()[0]
                                             if sum(g_mask_1) > 0 else "None")
                                            if binary_flag else active_nodes_num) if not tree_flag else ""
        visualize_network(
            weights=add_original_input(input_size, output_size, mlp_model.get_weights()) if tree_flag else _weights[0],
            acc=test_val_loss[1],
            comment=_comment)
        ### magnitudeプルーニング後のネットワーク構造を描画

        if tree_flag:  # 二分木mlpのうち不要ノードを削除(shrink)
            masked_mlp_model = masked_mlp(input_size, dense_size, output_size)
            show_weight(masked_mlp_model.get_weights())
            masked_mlp_model.set_weights(add_original_input(input_size, output_size, mlp_model.get_weights()))
            for target_layer in range(1, len(masked_mlp_model.get_weights()) // 2):
                print("shrink {}th layer".format(target_layer))
                masked_mlp_model = shrink_tree_nodes(masked_mlp_model, target_layer,
                                                     original_X_train, y_train, original_X_test, y_test,
                                                     only_active_list=False)

            ### 見やすいようにノードをソート
            sorted_weights = masked_mlp_model.get_weights()
            _mlp_shape = [sorted_weights[0].shape[0]] + \
                         [sorted_weights[2 * i + 1].shape[0] for i in range(len(sorted_weights) // 2)]
            _mask = [np.array([1 for _ in range(_mlp_shape[i])]) for i in range(len(_mlp_shape))]

            # visualize_network(sorted_weights, masked_mlp_model.evaluate(inputs_z(original_X_test, _mask), y_test)[1],
            # comment="shrinking all layer")

            # _mlp_model = mlp(_mlp_shape[0], _mlp_shape[1:-1], _mlp_shape[-1])
            # _mlp_model.set_weights(sorted_weights)
            # visualize_network(sorted_weights, acc=_mlp_model.evaluate(original_X_test, y_test)[1],
            # comment="not sorted any layers", non_active_neurons=None)
            for i in range(len(dense_size)):
                sorted_weights = sort_weights(sorted_weights, target_layer=i)
                masked_mlp_model.set_weights(sorted_weights)
                visualize_network(masked_mlp_model.get_weights(),
                                  masked_mlp_model.evaluate(inputs_z(original_X_test, _mask), y_test)[1],
                                  comment="sorted layer:{}".format(i))
                # _mlp_model.set_weights(sorted_weights)
                # print("_mlp_shape:{}".format(_mlp_shape))
                # visualize_network(sorted_weights, acc=_mlp_model.evaluate(original_X_test, y_test)[1],
                # comment="sorted layer:{}".format(i), non_active_neurons=None)
            ### 各層出力を可視化
            print("_mlp_shape:{}".format(_mlp_shape))
            # _mlp = masked_mlp(_mlp_shape[0], _mlp_shape[1:-1], _mlp_shape[-1])
            _mlp = mlp(_mlp_shape[0], _mlp_shape[1:-1], _mlp_shape[-1])
            _mlp.set_weights(sorted_weights)
            correct_data_train, correct_target_train, incorrect_data_train, incorrect_target_train \
                = show_intermidate_output(original_X_train, y_train, "train", _mlp)
            correct_data_test, correct_target_test, incorrect_data_test, incorrect_target_test \
                = show_intermidate_output(original_X_test, y_test, "test", _mlp)

            # show_intermidate_output(correct_data_train + correct_data_test + incorrect_data_train + incorrect_data_test,
            # correct_target_train + correct_target_test + incorrect_target_train + incorrect_target_test, "test", _mlp)
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            correct_data_test, correct_target_test,
                                            _mlp, name=["correct_train", "correct_test"])
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            incorrect_data_test, incorrect_target_test,
                                            _mlp, name=["correct_train", "miss_test"])
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            incorrect_data_train, incorrect_target_train,
                                            _mlp, name=["correct_train", "miss_train"])
            original_X_train, y_train = divide_data(original_X_train, y_train, dataset_category)
            original_X_test, y_test = divide_data(original_X_test, y_test, dataset_category)
            print("x_train:{}".format([len(i) for i in original_X_train]))
            print("x_test:{}".format([len(i) for i in original_X_test]))

            ###全結合mlpとの比較
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
                = getdata(dataset, binary_flag=binary_flag, train_frag=True)
            _mlp = mlp(_mlp_shape[0], _mlp_shape[1:-1], _mlp_shape[-1])
            _mlp.set_weights(sorted_weights)
            _mlp.fit(X_train, y_train, batch_size=cf.Minibatch, epochs=4000)
            correct_data_train, correct_target_train, incorrect_data_train, incorrect_target_train \
                = show_intermidate_output(X_train, y_train, "train", _mlp)
            correct_data_test, correct_target_test, incorrect_data_test, incorrect_target_test \
                = show_intermidate_output(X_test, y_test, "test", _mlp)
            original_X_train, original_X_test, y_train, y_test, train_num_per_step, data_inds, max_ite \
                = getdata(dataset, binary_flag=binary_flag, train_frag=False)
            visualize_network(_mlp.get_weights(), _mlp.evaluate(original_X_test, y_test)[1],
                              comment="mlp")
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            correct_data_test, correct_target_test,
                                            _mlp, name=["mlp_correct_train", "mlp_correct_test"])
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            incorrect_data_test, incorrect_target_test,
                                            _mlp, name=["mlp_correct_train", "mlp_miss_test"])
            show_intermidate_train_and_test(correct_data_train, correct_target_train,
                                            incorrect_data_train, incorrect_target_train,
                                            _mlp, name=["mlp_correct_train", "mlp_miss_train"])
            original_X_train, y_train = divide_data(original_X_train, y_train, dataset_category)
            original_X_test, y_test = divide_data(original_X_test, y_test, dataset_category)
            print("x_train:{}".format([len(i) for i in original_X_train]))
            print("x_test:{}".format([len(i) for i in original_X_test]))
            for i in range(dataset_category):
                print("mlp_train_acc:{}".format(_mlp.evaluate(original_X_train[i], y_train[i])[1]))
            for i in range(dataset_category):
                print("mlp_test_acc:{}".format(_mlp.evaluate(original_X_test[i], y_test[i])[1]))
                ###全結合mlpとの比較
        elif binary_flag:
            _X_test = [[] for _ in range(2)]
            _y_test = [[] for _ in range(2)]
            class_acc = [[] for _ in range(2)]
            for data, target in zip(X_test, y_test):
                _X_test[np.argmax(target)].append(data)
                _y_test[np.argmax(target)].append(target)
            _X_test = np.array(_X_test)
            _y_test = np.array(_y_test)
            for i in range(len(class_acc)):
                class_acc[i] = binary_classify.evaluate(inputs_z(_X_test[i], g_mask_1), np.array(_y_test[i]))
            for i in range(len(class_acc)):
                print("{}: {:0=5.2f}% <- {}sample".format(str(binary_target) + " " if i == 0 else "else",
                                                          class_acc[i][1] * 100, len(_y_test[i])))
        else:
            for target_layer in range(1, len(dense_size)):
                freezed_classify_1 = shrink_nodes(model=freezed_classify_1, target_layer=target_layer,
                                                  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            weights = freezed_classify_1.get_weights()
            architecture = [np.shape(weights[i * 2])[0] for i in range(len(weights) // 2)]
            g_mask_1 = [np.ones((architecture[i])) for i in range(len(architecture))]
            print("g_mask_1:{}".format([np.shape(i) for i in g_mask_1]))
            global dataset_category
            _X_test = [[] for _ in range(dataset_category)]
            _y_test = [[] for _ in range(dataset_category)]
            class_acc = [[] for _ in range(dataset_category)]
            for data, target in zip(X_test, y_test):
                _X_test[np.argmax(target)].append(data)
                _y_test[np.argmax(target)].append(target)
            for i in range(len(_X_test)):
                _X_test[i] = np.array(_X_test[i])
                _y_test[i] = np.array(_y_test[i])
            for i in range(len(class_acc)):
                class_acc[i] = freezed_classify_1.evaluate(inputs_z(_X_test[i], g_mask_1), _y_test[i])
            for i in range(len(class_acc)):
                print("{}: {:0=5.2f}% <- {}sample".format(i, class_acc[i][1] * 100, len(_y_test[i])))
            active_nodes = [[[] for __ in range(output_size)] for _ in range(len(weights) // 2)]
            for i in range(1, len(active_nodes)):
                active_nodes[i] = shrink_nodes(model=freezed_classify_1, target_layer=i,
                                               X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               only_active_list=True)
            for i in range(len(active_nodes)):
                print("active_nodes[{}]:{}".format(i, active_nodes[i]))
            ### クラスごとのactiveネットワーク構造を描画
            for i in range(output_size):
                print("\n\n\noutput_size:{}\nactive_nodes:{}".format(output_size, active_nodes_num))
                print("generating_architecture.....")
                im_architecture = active_route(copy.deepcopy(weights), acc=class_acc[i][1],
                                               comment="binary_target:{}".format(i), binary_target=i,
                                               using_nodes=[sum(active_nodes_num[:i]), active_nodes_num[i]],
                                               active_nodes=[_active[i] for _active in active_nodes])
                path = os.getcwd() + r"\visualized_iris\network_architecture\triple"
                # if not os.path.exists(path):
                # path = r"C:\Users\xeno\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture\triple"
                path += r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + "_{}.png".format(i))
                cv2.imwrite(path, im_architecture)
            ### クラスごとのactiveネットワーク構造を描画

            ### 各層の出力を描画
            im_input_train = show_result(input=X_train, onehot_labels=y_train,
                                         layer1_out=X_train,
                                         ite=cf.Iteration, classify=np.round(
                    freezed_classify_1.predict(inputs_z(X_train, g_mask_1), verbose=0)),
                                         testflag=False, showflag=True, comment="input")
            im_input_test = show_result(input=X_test, onehot_labels=y_test,
                                        layer1_out=X_test,
                                        ite=cf.Iteration, classify=np.round(
                    freezed_classify_1.predict(inputs_z(X_test, g_mask_1), verbose=0)),
                                        testflag=True, showflag=True, comment="input")
            im_g_dense_train = [[] for _ in range(len(hidden_layers))]
            im_g_dense_test = [[] for _ in range(len(hidden_layers))]

            _, _, _, _, hidden_layers, _, _ \
                = weightGAN_Model(input_size=input_size, wSize=dense_size[1], output_size=output_size, use_mbd=use_mbd,
                                  dense_size=dense_size)
            for i in range(len(hidden_layers)):
                hidden_layers[i].set_weights(freezed_classify_1.get_weights()[:i * 2 + 2])
            print("im_g_dense:{}".format(im_g_dense_train))
            for i in range(len(hidden_layers)):
                im_g_dense_train[i].append(show_result(input=X_train, onehot_labels=y_train,
                                                       layer1_out=hidden_layers[i].predict(inputs_z(X_train, g_mask_1),
                                                                                           verbose=0),
                                                       ite=cf.Iteration, classify=np.round(
                        freezed_classify_1.predict(inputs_z(X_train, g_mask_1), verbose=0)), testflag=False,
                                                       showflag=True, comment="dense{}".format(i)))
                im_g_dense_test[i].append(show_result(input=X_test, onehot_labels=y_test,
                                                      layer1_out=hidden_layers[i].predict(inputs_z(X_test, g_mask_1),
                                                                                          verbose=0),
                                                      ite=cf.Iteration, classify=np.round(
                        freezed_classify_1.predict(inputs_z(X_test, g_mask_1), verbose=0)), testflag=True,
                                                      showflag=True, comment="dense{}".format(i)))
            im_h_resize_train = im_input_train
            for im in im_g_dense_train:
                im_h_resize_train = hconcat_resize_min([im_h_resize_train, hconcat_resize_min(im)])
            im_h_resize_test = im_input_test
            for im in im_g_dense_test:
                im_h_resize_test = hconcat_resize_min([im_h_resize_test, hconcat_resize_min(im)])
            # im_h_resize = hconcat_resize_min([im_h_resize, np.array(im_architecture)])
            path = os.getcwd() + r"\visualized_iris\network_architecture\triple"
            cv2.imwrite(path + r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + ".png"), im_h_resize_train)
            cv2.imwrite(path + r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + ".png"), im_h_resize_test)
            ### 各層の出力を描画


def save_images(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch = imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num * H, w_num * W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y * H:(y + 1) * H, x * W:(x + 1) * W] = batch[i, ..., 0]
    fname = str(index).zfill(len(str(cf.Iteration))) + '.jpg'
    save_path = os.path.join(dir_path, fname)

    if cf.Save_iteration_disp:
        plt.imshow(out, cmap='gray')
        plt.title("iteration: {}".format(index))
        plt.axis("off")
        plt.savefig(save_path)
    else:
        cv2.imwrite(save_path, out)


def save_images_separate(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch = imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    for i in range(B):
        save_path = os.path.join(dir_path, '{}_{}.jpg'.format(index, i))
        cv2.imwrite(save_path, batch[i])


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--test_non_trained', dest='test_non_trained', action='store_true')
    parser.add_argument('--mnist', dest='mnist', action='store_true')
    parser.add_argument('--cifar10', dest='cifar10', action='store_true')
    parser.add_argument('--iris', dest='iris', action='store_true')
    parser.add_argument('--wine', dest='wine', action='store_true')
    parser.add_argument('--balance', dest='balance', action='store_true')
    parser.add_argument('--digits', dest='digits', action='store_true')
    parser.add_argument('--krkopt', dest='krkopt', action='store_true')
    parser.add_argument('--use_mbd', dest='use_mbd', action='store_true')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--no_mask', dest='no_mask', action='store_true')
    parser.add_argument('--load_model', dest='load_model', action='store_true')
    # parser.add_argument('--pruning', dest='pruning', action='store_true')
    parser.add_argument('--pruning_rate', type=float, default=-1)
    parser.add_argument('--wSize', type=int)
    parser.add_argument('--binary_target', type=int, default=-1)
    parser.add_argument('--tree', dest='tree', action='store_true')
    parser.add_argument('--parity', dest='parity', action='store_true')
    parser.add_argument('--parity_shape', type=int)
    parser.add_argument('--child_num', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    use_mbd = False
    if args.tree:
        tree_flag = True
    if args.child_num:
        CHILD_NUM = args.child_num
    if args.wSize:
        wSize = args.wSize
    if args.no_mask:
        no_mask = True
    if args.use_mbd:
        use_mbd = True
    if args.pruning_rate >= 0:
        print("args.pruning_rate:{}".format(args.pruning_rate))
        pruning_rate = args.pruning_rate
    if args.binary:
        binary_flag = True
    if args.mnist:
        Height = 28
        Width = 28
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 10
        from keras.datasets import mnist
        import config_mnist as cf
        from _model_mnist import *

        # np.random.seed(cf.Random_seed)
        dataset = "mnist"
    elif args.cifar10:
        Height = 32
        Width = 32
        Channel = 3
        input_size = Height * Width * Channel
        output_size = 10
        # import config_cifar10 as cf
        import config_mnist as cf
        from keras.datasets import cifar10 as mnist
        # from _model_cifar10 import *
        from _model_mnist import *

        # np.random.seed(cf.Random_seed)
        dataset = "cifar10"
    elif args.iris:
        Height = 1
        Width = 4
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 3
        dataset_category = 3
        import config_mnist as cf

        cf.Dataset = "iris_"
        from _model_iris import *
        # np.random.seed(cf.Random_seed)
        from sklearn.datasets import load_iris

        iris = load_iris()
        dataset = "iris"
    elif args.digits:
        Height = 1
        Width = 64
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 10
        import config_mnist as cf

        cf.Dataset = "digits_"
        from _model_iris import *
        # np.random.seed(cf.Random_seed)
        # from sklearn.datasets import load_iris
        from sklearn.datasets import load_digits

        dataset_category = 10
        digits = load_digits(n_class=dataset_category)
        dataset = "digits"
    elif args.krkopt:
        Height = 1
        Width = 6
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 17
        import config_mnist as cf
        from _model_iris import *

        # np.random.seed(cf.Random_seed)
        # from sklearn.datasets import load_iris
        # iris = load_iris()
        dataset = "krkoptflag"
    elif args.wine:
        Height = 1
        Width = 13
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 3
        dataset_category = 3
        import config_mnist as cf

        cf.Dataset = "wine_"
        from _model_iris import *

        dataset = "wine"
    elif args.balance:
        Height = 1
        Width = 4
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 3
        dataset_category = 3
        import config_mnist as cf
        from _model_iris import *

        dataset = "balance"
    elif args.parity:
        Height = 1
        Width = 16
        if args.parity_shape:
            Width = args.parity_shape
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 2
        dataset_category = 2
        import config_mnist as cf
        from _model_iris import *

        dataset = "parity"
    dense_size[0] = input_size
    if args.binary_target >= 0:
        binary_flag = True
        binary_target = args.binary_target
        cf.Save_binary_classify_path = cf.Save_binary_classify_path[:-3] \
                                       + str(binary_target) + \
                                       cf.Save_binary_classify_path[-3:]
        for i in range(len(cf.Save_hidden_layers_path)):
            _path = cf.Save_hidden_layers_path[i]
            cf.Save_hidden_layers_path[i] = _path[:-3] + "_" + str(binary_target) + _path[-3:]
            print(_path)
    cf.reload_path()
    if args.train:
        main = Main_train()
        main.train(use_mbd=use_mbd, load_model=args.load_model)
    if args.test:
        main = Main_test()
        main.test()
    if args.test_non_trained:
        main = Main_test()
        main.test(loadflag=False)
    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")

#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.layers.core import Lambda

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import argparse
import cv2
import numpy as np
import copy

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=2000)
import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from visualization import visualize, hconcat_resize_min, vconcat_resize_min
from _model_weightGAN import weightGAN_Model
# import config_mnist as cf

# from _model_mnist import *
### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from draw_architecture import *

### モデル量子化
from keras.models import load_model
from keras_compressor.compressor import compress
### モデル量子化

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils

Height, Width = 28, 28
Channel = 1
output_size = 10
input_size = Height * Width * Channel
wSize = 20
dataset = ""
binary_flag = False
binary_target = -1
pruning_rate=-1

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
    y_train = np_utils.to_categorical(y_train, 3)
    y_test = np_utils.to_categorical(y_test, 3)
    train_num = X_train.shape[0]
    train_num_per_step = train_num // cf.Minibatch
    data_inds = np.arange(train_num)
    max_ite = cf.Minibatch * train_num_per_step
    print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite

def stringToList(_string, split=" "):
    str_list = []
    temp = ''
    for x in _string:
        if x == split:  # 区切り文字
            str_list.append(temp)
            temp = ''
        else:
            temp += x
    if temp != '':  # 最後に残った文字列を末尾要素としてリストに追加
        str_list.append(temp)
    return str_list

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

def wine_data():
    ### .dataファイルを読み込む
    _data = open(r"C:\Users\papap\Documents\research\DCGAN_keras-master\wine.data", "r")
    lines = _data.readlines()
    ### .dataファイルを読み込む

    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む
    _train = []
    _target = []
    for line in lines:
        line = stringToList(line, ",")
        _target.append(int(line[0])-1) #今回はリストの先頭がクラスラベル(1 or 2 or 3)
        _train.append([float(i) for i in line[1:]]) #それ以外は訓練データ
    ### .dataファイルから","をsplitとして、1行ずつリストとして読み込む

    X_train, X_test, y_train, y_test = \
        train_test_split(np.array(_train), np.array(_target), test_size=0.1, train_size=0.9, shuffle=True, random_state=1)

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
    print("X_train:\n{} \nX_test:\n{}".format(X_train, X_test))
    print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
    print("-> X_max in train :{}".format(np.amax(X_train, axis=0)))
    print("-> X_max in test  : {}".format(np.amax(X_test, axis=0)))
    # print("X_train:\n{}".format(X_train))
    # exit()
    return X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite

def digits_data():
    usable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]# random.sample([int(i) for i in range(10)], 2)
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
    X_train, X_test = normalize(X_train, X_test)
    print("X_train:{}".format(digits.data.shape))
    print("X_train:{}".format(X_train.shape))
    print("y_train:{}".format(digits.target.shape))
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
            for j in range(9):
                y_train.append([1])
                X_train.append(_X_train[i])
        else:
            # _y_train[i] = 0
            y_train.append([0])
            X_train.append(_X_train[i])
    for i in range(len(_y_test)):
        if _y_test[i] == usable[binary_target]:
            # _y_test[i] = 1
            for j in range(9):
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

def getdata(dataset):
    if dataset == "iris":
        return iris_data()
    elif  dataset == "mnist":
        return mnist_data()
    elif dataset == "digits":
        return digits_data()
    elif dataset == "wine":
        return wine_data()

def mask(_mask, batch_size):
    return np.array([_mask for _ in range(batch_size)])

def show_weight(weights):
    for i in range(len(weights)):
        print("\n{}\n".format(np.shape(weights[i])))
        print(weights[i])

def load_concate_masks(active_true=False):
    ### 保存されているmaskにおける、and集合を返す
    g_mask = np.zeros(wSize)
    for i in range(10):
        _g_mask = np.load(cf.Save_layer_mask_path[i])
        # print("g_mask:{}\n{}".format(g_mask.shape, g_mask))
        # print("_g_mask:{}\n{}".format(_g_mask.shape, _g_mask))
        for j in range(g_mask.shape[0]):
            if g_mask[j] == 1 or _g_mask[j] == 1:
                g_mask[j] = 1
    if not active_true:
        for i in range(len(g_mask)):
            g_mask[i] = (g_mask[i] + 1) % 2
    return g_mask


class Main_train():
    def __init__(self):
        pass

    def train(self, load_model=False, use_mbd=False):
        # 性能評価用パラメータ
        max_score = 0.
        if binary_flag:
            if binary_target == 0:
                # active nodesなしで初期化
                g_mask_1 = np.zeros(wSize)
                ### layer_maskファイルを全初期化
                for i in range(10):
                    np.save(cf.Save_layer_mask_path[i], g_mask_1)
                g_mask_1 = load_concate_masks(active_true=False)# [1,1,1,1,...1]
            else:
                g_mask_1 = load_concate_masks(active_true=False)
        else:
            g_mask_1 = load_concate_masks(active_true=True)# np.load(cf.Save_layer_mask_path)
            # g_mask_1 = np.ones(wSize)
        print("wSize:{}".format(wSize))
        g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1\
            = weightGAN_Model(input_size=input_size, wSize=wSize, output_size=output_size, use_mbd=use_mbd)
        if load_model:
            freezed_classify_1.save_weights(cf.Save_freezed_classify_1_path)
            g.load_weights(cf.Save_g_path)
            d.load_weights(cf.Save_d_path)
            c.load_weights(cf.Save_c_path)
            classify.load_weights(cf.Save_classify_path)
            print("classify.summary()")
            classify.summary()
            print("binary_classify.summary()")
            binary_classify.summary()
            binary_classify.load_weights(cf.Save_binary_classify_path)
            if binary_flag:
                for i in range(len(hidden_layers)):
                    print("hiddden_layers[{}].summary()".format(i))
                    hidden_layers[i].summary()
                    hidden_layers[i].load_weights(cf.Save_hidden_layers_path[i])
            else:
                ### [0,1,...,9]の重みとバイアス(入力層->中間層)を読み込む
                syncro_weights = [np.zeros((input_size, wSize)), np.zeros((wSize, ))]
                for i in range(10):
                    binary_classify.load_weights(cf.Save_binary_classify_path[:-3] + str(i) + cf.Save_binary_classify_path[-3:])
                    syncro_weights[0] = np.add(syncro_weights[0], binary_classify.get_weights()[0])
                    syncro_weights[1] = np.add(syncro_weights[1], binary_classify.get_weights()[1])
                print("hiddden_layers[{}].summary()".format(0))
                # hidden_layers[0].set_weights(syncro_weights)
                # hidden_layers[0].save_weights(cf.Save_hidden_layers_path[0])
                freezed_classify_1.load_weights(cf.Save_freezed_classify_1_path)
                freezed_classify_1.set_weights(syncro_weights+(freezed_classify_1.get_weights()[2:]))
                freezed_classify_1.save_weights(cf.Save_freezed_classify_1_path)
                freezed_classify_1.load_weights(cf.Save_freezed_classify_1_path)
                show_weight(freezed_classify_1.get_weights())
                im_architecture = mydraw(freezed_classify_1.get_weights(), -1,
                                         comment="using all classes\nsyncro graph\n{}".format(g_mask_1))
                ### ネットワーク構造を描画
                im_h_resize = im_architecture
                path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture\triple" \
                       + r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
                cv2.imwrite(path, im_h_resize)
                ### [0,1,...,9]の重みとバイアス(入力層->中間層)を読み込む

        ### Prepare Training data　前処理
        if dataset == "iris":
            fname = os.path.join(cf.Save_dir, 'loss_iris.txt')
        elif dataset == "digits":
            fname = os.path.join(cf.Save_dir, 'loss_digits.txt')
        elif dataset == "krkoptflag":
            fname = os.path.join(cf.Save_dir, 'loss_krkopt.txt')
        else:
            fname = os.path.join(cf.Save_dir, 'loss.txt')
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = getdata(dataset)
        """
        ## Start Train
        print('-- Training Start!!')
        if cf.Save_train_combine is None:
            print("generated image will not be stored")
        elif cf.Save_train_combine is True:
            print("generated image write combined >>", cf.Save_train_img_dir)
        elif cf.Save_train_combine is False:
            print("generated image write separately >>", cf.Save_train_img_dir)
        """
        f = open(fname, 'w')
        f.write("Iteration,G_loss,D_loss{}".format(os.linesep))
        ### Prepare Training data　前処理

        for ite in range(cf.Iteration):
            ite += 1
            train_ind = ite % train_num_per_step
            if ite % (train_num_per_step + 1) == 0:
                np.random.shuffle(data_inds)
            _inds = data_inds[train_ind * cf.Minibatch: (train_ind + 1) * cf.Minibatch]

            ### GAN用の真画像(real_weight)、分類ラベル(real_labels)生成
            real_weight = np.zeros((cf.Minibatch, wSize))
            real_labels = np.zeros((cf.Minibatch, output_size))
            """
            for i in range(cf.Minibatch):
                _list = list(range(wSize))
                random.shuffle(_list)
                for j in _list[:2]: # ランダムに4つ選んで発火するようノード選択
                    real_weight[i][j] = 1
                real_labels[i][np.random.randint(output_size)] = 1 # ノードごとの真の画像ラベル条件
                if y_train[_inds][i][0] == 1:
                    real_weight[i][0] = 1
                    # real_weight[i][1] = 1
                    real_labels[i][0] = 1
                    # real_weight[i][2] = 1
                elif y_train[_inds][i][1] == 1:
                    real_weight[i][2] = 1
                    # real_weight[i][4] = 1
                    # real_weight[i][5] = 1
                    real_labels[i][1] = 1
                elif y_train[_inds][i][2] == 1:
                    real_weight[i][4] = 1
                    # real_weight[i][7] = 1
                    # real_weight[i][8] = 1
                    real_labels[i][2] = 1
                """
            ### GAN用のreal画像、分類ラベル生成

            ### GAN用のfake画像、分類ラベル生成
            fake_weight = g.predict([X_train[_inds], mask(g_mask_1, cf.Minibatch)], verbose=0)[1]
            fake_labels = y_train[_inds]
            ### GAN用のfake画像、分類ラベル生成

            t = np.array([1] * cf.Minibatch + [0] * cf.Minibatch)
            # concated_weight = np.concatenate((fake_weight, real_weight))
            # concated_labels = np.concatenate((fake_labels, real_labels))

            # if ite % 100 == 0:
                # d_loss = d.train_on_batch([concated_weight, concated_labels], t)  # これで重み更新までされる
            # else:
            d_loss = 0

            t = np.array([0] * cf.Minibatch)
            # g_loss = 0 # c.train_on_batch([X_train[_inds], y_train[_inds]], [y_train[_inds], t])  # 生成器を学習
            if binary_flag:
                g_loss = binary_classify.train_on_batch([X_train[_inds], mask(g_mask_1, cf.Minibatch)], y_train[_inds])  # 生成器を学習
            else:
                g_loss = freezed_classify_1.train_on_batch([X_train[_inds], mask(g_mask_1, cf.Minibatch)], y_train[_inds])
                # g_loss = c.train_on_batch([X_train[_inds], y_train[_inds], mask(g_mask_1, cf.Minibatch)], [y_train[_inds], t])  # 生成器を学習
            con = my_tqdm(ite)
            if ite % cf.Save_train_step == 0:
                if binary_flag:
                    test_val_loss = binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
                    train_val_loss = binary_classify.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
                else:
                    # test_val_loss = classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
                    # train_val_loss = classify.evaluate([X_train,mask(g_mask_1 ,len(X_train))], y_train)
                    test_val_loss = freezed_classify_1.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
                    train_val_loss = freezed_classify_1.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
                max_score = max(max_score, test_val_loss[1])
                # if binary_flag:
                con += "Ite:{}, catego: loss{:.6f} acc:{:.6f} g: {:.6f}, d: {:.6f}, test_val: loss:{:.6f} acc:{:.6f}".format(
                    ite, g_loss[0], train_val_loss[1], g_loss[1], d_loss, test_val_loss[0], test_val_loss[1])
                # else:
                    # con += "Ite:{}, catego: loss{:.6f} acc:{:.6f} g: {:.6f}, d: {:.6f}, test_val: loss:{:.6f} acc:{:.6f}".format(
                        # ite, g_loss[1], train_val_loss[1], g_loss[2], d_loss, test_val_loss[0], test_val_loss[1])
                # print("real_weight\n{}".format(real_weight))
                # print("real_labels\n{}".format(real_labels))
                # print("layer1_out:train\n{}".format(np.round(fake_weight, decimals=2)))  # 訓練データ時
                if ite % cf.Save_train_step == 0:
                    if dataset == "iris":
                        # print("labels:{}".format(np.argmax(y_train, axis=1)))
                        show_result(input=X_train, onehot_labels=y_train,
                                    layer1_out=np.round(g.predict([X_train, mask(g_mask_1, len(X_train))], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict([X_train, mask(g_mask_1, len(X_train))], verbose=0)[0], decimals=2), testflag=False)
                        show_result(input=X_test, onehot_labels=y_test,
                                    layer1_out=np.round(g.predict([X_test, mask(g_mask_1, len(X_test))], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict([X_test, mask(g_mask_1, len(X_test))], verbose=0)[0], decimals=2), testflag=True)
                    """
                    else:
                        show_result(input=X_train[:100], onehot_labels=y_train[:100],
                                    layer1_out=np.round(g.predict(X_train[:100], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_train[:100], verbose=0)[0], decimals=2),
                                    testflag=False)
                        show_result(input=X_test[:100], onehot_labels=y_test[:100],
                                    layer1_out=np.round(g.predict(X_test[:100], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_test[:100], verbose=0)[0], decimals=2), testflag=True)
                    """
            else:
                # if binary_flag:
                con += "Ite:{}, catego:{} g:{}, d: {:.6f}".format(ite, g_loss[0], g_loss[1], d_loss)
                # else:
                    # con += "Ite:{}, catego:{} g:{}, d: {:.6f}".format(ite, g_loss[1], g_loss[2], d_loss)

            sys.stdout.write("\r" + con)

            if ite % cf.Save_train_step == 0 or ite == 1:
                print()
                f.write("{},{},{}{}".format(ite, g_loss, d_loss, os.linesep))
                # save weights
                if binary_flag:
                    d.save_weights(cf.Save_d_path)
                    g.save_weights(cf.Save_g_path)
                    c.save_weights(cf.Save_c_path)
                    classify.save_weights(cf.Save_classify_path)
                    binary_classify.save_weights(cf.Save_binary_classify_path)
                    for i in range(len(hidden_layers)):
                        hidden_layers[i].save_weights(cf.Save_hidden_layers_path[i])
                    # np.save(cf.Save_layer_mask_path[binary_target], g_mask_1)
                else:
                    freezed_classify_1.save(cf.Save_freezed_classify_1_path)
                """
                gerated = g.predict([z], verbose=0)
                # save some samples
                if cf.Save_train_combine is True:
                    save_images(gerated, index=str(ite)+" loss:{}".format(cnn_val_loss), dir_path=cf.Save_train_img_dir)
                elif cf.Save_train_combine is False:
                    save_images_separate(gerated, index=str(ite)+" loss:{}".format(cnn_val_loss), dir_path=cf.Save_train_img_dir)
                """
        f.close()
        ## Save trained model
        if binary_flag:
            d.save_weights(cf.Save_d_path)
            g.save_weights(cf.Save_g_path)
            c.save_weights(cf.Save_c_path)
            classify.save_weights(cf.Save_classify_path)
            binary_classify.save_weights(cf.Save_binary_classify_path)
            for i in range(len(hidden_layers)):
                hidden_layers[i].save_weights(cf.Save_hidden_layers_path[i])
        else:
            freezed_classify_1.save(cf.Save_freezed_classify_1_path)
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path, cf.Save_classify_path)
        print("maxAcc:{}".format(max_score * 100))
        ### add for TensorBoard
        KTF.set_session(old_session)
        ###


def show_result(input, onehot_labels, layer1_out, ite, classify, testflag=False, showflag=False, comment=""):
    print("\n{}".format(" test" if testflag else "train"))
    labels_scalar = np.argmax(onehot_labels, axis=1)
    # print("labels:{}".format(labels_scalar))
    layer1_outs = [[[], [], []] for _ in range(output_size)]
    for i in range(len(labels_scalar)):
        layer1_outs[labels_scalar[i]][0].append(input[i])  # 入力
        layer1_outs[labels_scalar[i]][1].append(layer1_out[i])  # 中間層出力
        layer1_outs[labels_scalar[i]][2].append(classify[i])  # 分類層出力
    # print("layer1_outs:{}".format(layer1_outs))
    for i in range(len(layer1_outs)):
        print("\nlabel:{}".format(i))
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
    return visualize([_outs[1] for _outs in layer1_outs], [_outs[2] for _outs in layer1_outs], labels_scalar, ite,
                     testflag, showflag=showflag, comment=comment)
    # visualize([_outs[1] for _outs in layer1_outs], [_outs[2] for _outs in layer1_outs], labels_scalar, ite,
    # testflag, showflag=False)


class Main_test():
    def __init__(self):
        pass

    def test(self, loadflag=True):
        ite = 0
        X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = getdata(dataset)
        g, d, c, classify, hidden_layers, binary_classify, freezed_classify_1\
            = weightGAN_Model(input_size=input_size, wSize=wSize, output_size=output_size, use_mbd=use_mbd)
        g_mask_1 = load_concate_masks(active_true=(not binary_flag))# np.load(cf.Save_layer_mask_path)
        print("g_mask:{}".format(g_mask_1))
        """
        if not binary_flag:
            for i in range(len(g_mask_1)):
                g_mask_1[i] = (g_mask_1[i] + 1) % 2
            g_mask_1 = np.ones(wSize)
        """
        if loadflag:
            if binary_flag:
                g.load_weights(cf.Save_g_path)
                d.load_weights(cf.Save_d_path)
                c.load_weights(cf.Save_c_path)
                classify.load_weights(cf.Save_classify_path)
                binary_classify.load_weights(cf.Save_binary_classify_path)
                print("load:{}".format(cf.Save_binary_classify_path))
                # exit()
                for i in range(len(hidden_layers)):
                    hidden_layers[i].load_weights(cf.Save_hidden_layers_path[i])
                # g = compress(g, 7e-1)
                # d = compress(d, 7e-1)
                # c = compress(c, 7e-1)
                # classify = compress(classify, 7e-1)
                # for i in range(len(hidden_layers)):
                    # hidden_layers[i] = compress(hidden_layers[i], 7e-1)
                _weights = [binary_classify.get_weights(), binary_classify.get_weights()]
            else:
                freezed_classify_1.load_weights(cf.Save_freezed_classify_1_path)
                _weights = [freezed_classify_1.get_weights(), freezed_classify_1.get_weights()]
            # _weights[高精度確定重み, プルーニング重み]

            # print("_weights\n{}".format(_weights))
            if binary_flag:
                test_val_loss = binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)  # [0.026, 1.0]
                # train_val_loss = binary_classify.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
                pruned_test_val_loss = copy.deepcopy(test_val_loss)
                # pruned_train_val_loss = copy.deepcopy(train_val_loss)
            else:
                # test_val_loss = classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
                # train_val_loss = classify.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
                test_val_loss = freezed_classify_1.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
                pruned_test_val_loss = copy.deepcopy(test_val_loss)
                # pruned_train_val_loss = copy.deepcopy(train_val_loss)
            print("pruned_test_val_loss:{}".format(pruned_test_val_loss))
            # exit()
            ### 重みプルーニング
            global pruning_rate
            print("pruning_rate:{}".format(pruning_rate))
            if pruning_rate >= 0:
                while (pruned_test_val_loss[1] > test_val_loss[1] * 0.95) and pruning_rate < 1:
                    ### 精度98%以上となる重みを_weights[0]に確保
                    _weights[0] = copy.deepcopy(_weights[1])
                    ### 精度98%以上となる重みを_weights[0]に確保

                    ### プルーニング率を微上昇させ性能検証
                    pruning_rate += 0.01
                    for i in range(np.shape(_weights[1])[0]):
                        if _weights[1][i].ndim == 2:
                            print("np.shape(_weights[1][{}]):{}".format(i, np.shape(_weights[1][i])))
                            for j in range(np.shape(_weights[1][i])[0]):
                                for k in range(np.shape(_weights[1][i])[1]):
                                    if abs(_weights[1][i][j][k]) < pruning_rate:
                                        _weights[1][i][j][k] = 0.
                            print("weights[{}]:{} (>0)".format(i, np.count_nonzero(_weights[1][i] > 0)))
                    if binary_flag:
                        binary_classify.set_weights(_weights[1])
                        pruned_test_val_loss = binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)  # [0.026, 1.0]
                    else:
                        freezed_classify_1.set_weights(_weights[1])
                        pruned_test_val_loss = freezed_classify_1.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
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

            ### 第1中間層ノードプルーニング
            active_nodes = []
            acc_list = []
            if binary_flag:
                g_mask_1 = load_concate_masks(False) # activeでないノード=1
                ### 欠落させると精度が落ちるノードを検出
                pruned_test_val_acc = \
                    binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)[1]
                for i in range(wSize):
                    # g_mask_1 = np.ones(wSize)
                    g_mask_1[i] = 0
                    _acc = \
                        binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)[1]  # [0.026, 1.0]
                    acc_list.append([_acc, i])
                    if _acc < pruned_test_val_acc * 0.999:
                        active_nodes.append(i)
                    g_mask_1[i] = 1
                # g_mask_1 = load_concate_masks(active_true=True) # np.load(cf.Save_layer_mask_path)
                # for i in active_nodes:
                    # g_mask_1[i] = 0 # active_nodeは使用中フラグを立てる
                if len(active_nodes)==0:
                    active_nodes.append(sorted(acc_list)[-1][1])
                    # g_mask_1[sorted(acc_list)[-1][1]] = 0
                concated_active = np.zeros(wSize)
                for i in active_nodes:
                    concated_active[i] = 1
                np.save(cf.Save_layer_mask_path[binary_target], concated_active)# g_mask_1)
                # g_mask_1の内、占有ノードのみ使用できるようにする
                # [0, 1, ...binary_target]でactiveとなったノードの集合
                g_mask_1 = concated_active# load_concate_masks(active_true=True)
                # for i in range(len(g_mask_1)):
                    # g_mask_1[i] = (g_mask_1[i] + 1) % 2
                # g_mask_1 = list(np.bitwise_not(np.array(g_mask_1)))
                print("active_nodes:{}".format(active_nodes))
            ### 第1中間層ノードプルーニング
            else:
                active_nodes = [-1]# g_mask_1

        if (not load_model) and (pruning_rate >= 0):
            print("\nError : Please load Model to do pruning")
            exit()
        # t = np.array([0] * len(X_train))
        # g_loss = c.train_on_batch([X_train, y_train], [y_train, t])  # 生成器を学習
        if binary_flag:
            test_val_loss = binary_classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test) # [0.026, 1.0]
            train_val_loss = binary_classify.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
            # binary_classify.load_weights(cf.Save_binary_classify_path)
            weights = binary_classify.get_weights()# classify.get_weights()
        else:
            # test_val_loss = classify.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
            # train_val_loss = classify.evaluate([X_train, mask(g_mask_1, len(X_train))], y_train)
            # weights = classify.get_weights()
            test_val_loss = freezed_classify_1.evaluate([X_test, mask(g_mask_1, len(X_test))], y_test)
            weights = freezed_classify_1.get_weights()
        """
        if not binary_flag:
            im_input_train = show_result(input=X_train, onehot_labels=y_train,
                                         layer1_out=X_train,
                                         ite=cf.Iteration, classify=np.round(g.predict([X_train, mask(g_mask_1, len(X_train))], verbose=0)[0]), testflag=False,
                                         showflag=True, comment="input")
            im_input_test = show_result(input=X_test, onehot_labels=y_test,
                                         layer1_out=X_test,
                                         ite=cf.Iteration, classify=np.round(g.predict([X_train, mask(g_mask_1, len(X_train))], verbose=0)[0]),
                                         testflag=True, showflag=True, comment="input")
            im_g_dense = [[] for _ in range(len(hidden_layers))]
            print("im_g_dense:{}".format(im_g_dense))
            for i in range(len(hidden_layers)):
                im_g_dense[i].append(show_result(input=X_train, onehot_labels=y_train,
                                                 layer1_out=hidden_layers[i].predict([X_train, mask(g_mask_1, len(X_train))], verbose=0),
                                                 ite=cf.Iteration, classify=np.round(g.predict([X_train, mask(g_mask_1, len(X_train))], verbose=0)[0]), testflag=False,
                                                 showflag=True, comment="dense{}".format(i)))
                im_g_dense[i].append(show_result(input=X_test, onehot_labels=y_test,
                                                 layer1_out=hidden_layers[i].predict([X_test, mask(g_mask_1, len(X_test))], verbose=0),
                                                 ite=cf.Iteration, classify=np.round(g.predict([X_test, mask(g_mask_1, len(X_test))], verbose=0)[0]), testflag=True,
                                                 showflag=True, comment="dense{}".format(i)))
            print("Ite:{}, train: loss :{:.6f} acc:{:.6f} test_val: loss:{:.6f} acc:{:.6f}"
                  .format(ite, train_val_loss[0], train_val_loss[1], test_val_loss[0], test_val_loss[1]))
        """

        for i in range(len(weights)):
            print(np.shape(weights[i]))
        classify.summary()

        print("\ntest Acc:{}".format(test_val_loss[1]))
        print("g_mask_1\n{}".format(g_mask_1))
        if binary_flag:
            print("g_mask_in_binary\n{}".format(load_concate_masks(active_true=True)))

        ### ネットワーク構造を描画
        im_architecture = mydraw(weights, test_val_loss[1],
                                 comment=("[{} vs other]".format(binary_target) if binary_flag else "")
                                         + " pruned <{:.4f}\n".format(pruning_rate)
                                         + "active_node:{}".format(active_nodes if len(active_nodes)>0 else "None"))
        ### ネットワーク構造を描画
        im_h_resize = im_architecture
        """
        if not binary_flag:
            im_h_resize = hconcat_resize_min([im_input_train, im_input_test])
            for im in im_g_dense:
                # im_h_resize = vconcat_resize_min([hconcat_resize_min([np.array(im_train), np.array(im_test)]), im_h_resize])
                im_h_resize = vconcat_resize_min([hconcat_resize_min(im), im_h_resize])
            im_h_resize = hconcat_resize_min([im_h_resize, np.array(im_architecture)])
        """
        path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture\triple"\
               + r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
        cv2.imwrite(path, im_h_resize)
        print("saved concated graph to -> {}".format(path))
    def _test(self):
        ## Load network model
        g = G_model(Height=Height, Width=Width, channel=Channel)
        g.load_weights(cf.Save_g_path, by_name=True)

        print('-- Test start!!')
        if cf.Save_test_combine is None:
            print("generated image will not be stored")
        elif cf.Save_test_combine is True:
            print("generated image write combined >>", cf.Save_test_img_dir)
        elif cf.Save_test_combine is False:
            print("generated image write separately >>", cf.Save_test_img_dir)
        pbar = tqdm(total=cf.Test_num)

        for i in range(cf.Test_num):
            input_noise = np.random.uniform(-1, 1, size=(cf.Test_Minibatch, 100))
            g_output = g.predict(input_noise, verbose=0)

            if cf.Save_test_combine is True:
                save_images(g_output, index=i, dir_path=cf.Save_test_img_dir)
            elif cf.Save_test_combine is False:
                save_images_separate(g_output, index=i, dir_path=cf.Save_test_img_dir)
            pbar.update(1)


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
    parser.add_argument('--digits', dest='digits', action='store_true')
    parser.add_argument('--krkopt', dest='krkopt', action='store_true')
    parser.add_argument('--use_mbd', dest='use_mbd', action='store_true')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--load_model', dest='load_model', action='store_true')
    # parser.add_argument('--pruning', dest='pruning', action='store_true')
    parser.add_argument('--pruning_rate', type=float, default=-1)
    parser.add_argument('--wSize', type=int)
    parser.add_argument('--binary_target', type=int, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    use_mbd = False
    if args.wSize:
        wSize = args.wSize
    if args.use_mbd:
        use_mbd = True
    if args.pruning_rate >= 0:
        print("args.pruning_rate:{}".format(args.pruning_rate))
        pruning_rate = args.pruning_rate
    if args.binary or (args.binary_target >= 0):
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
        import config_cifar10 as cf
        from keras.datasets import cifar10 as mnist
        from _model_cifar10 import *
        # np.random.seed(cf.Random_seed)
        dataset = "cifar10"
    elif args.iris:
        Height = 1
        Width = 4
        Channel = 1
        input_size = Height * Width * Channel
        output_size = 3
        import config_mnist as cf
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
        from _model_iris import *
        # np.random.seed(cf.Random_seed)
        # from sklearn.datasets import load_iris
        from sklearn.datasets import load_digits
        digits = load_digits(n_class=10)
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
        import config_mnist as cf
        from _model_iris import *
        dataset = "wine"

    if args.binary_target >= 0:
        binary_flag = True
        binary_target = args.binary_target
        cf.Save_binary_classify_path = cf.Save_binary_classify_path[:-3]\
                                       +str(binary_target)+\
                                       cf.Save_binary_classify_path[-3:]
        for i in range(len(cf.Save_hidden_layers_path)):
            _path = cf.Save_hidden_layers_path[i]
            cf.Save_hidden_layers_path[i] = _path[:-3] + "_" +str(binary_target) + _path[-3:]
            print(_path)
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

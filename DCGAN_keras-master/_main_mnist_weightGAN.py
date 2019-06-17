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
irisflag = False
krkoptflag = False
wSize = 20


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


class Main_train():
    def __init__(self):
        pass

    def train(self, use_mbd=False):
        print("irisflag:{}".format(irisflag))
        # 性能評価用パラメータ
        max_score = 0.
        g, d, c, classify = weightGAN_Model(input_size=input_size, wSize=wSize, output_size=output_size,
                                            use_mbd=use_mbd)
        ## Prepare Training data　前処理
        if irisflag:
            fname = os.path.join(cf.Save_dir, 'loss_iris.txt')
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = iris_data()
        elif krkoptflag:
            fname = os.path.join(cf.Save_dir, 'loss_krkopt.txt')
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = krkopt_data()
        else:
            fname = os.path.join(cf.Save_dir, 'loss.txt')
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = mnist_data()

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
            fake_weight = g.predict(X_train[_inds], verbose=0)[1]
            fake_labels = y_train[_inds]
            ### GAN用のfake画像、分類ラベル生成

            t = np.array([1] * cf.Minibatch + [0] * cf.Minibatch)
            concated_weight = np.concatenate((fake_weight, real_weight))
            concated_labels = np.concatenate((fake_labels, real_labels))

            if ite % 100 == 0:
                d_loss = d.train_on_batch([concated_weight, concated_labels], t)  # これで重み更新までされる
            else:
                d_loss = 0

            t = np.array([0] * cf.Minibatch)
            g_loss = c.train_on_batch([X_train[_inds], y_train[_inds]], [y_train[_inds], t])  # 生成器を学習

            con = my_tqdm(ite)
            if ite % cf.Save_train_step == 0:
                test_val_loss = classify.evaluate(X_test, y_test)
                train_val_loss = classify.evaluate(X_train, y_train)
                max_score = max(max_score, 1. - test_val_loss[1])
                con += "Ite:{}, catego: loss{:.6f} acc:{:.6f} g: {:.6f}, d: {:.6f}, test_val: loss:{:.6f} acc:{:.6f}".format(
                    ite, g_loss[1], train_val_loss[1], g_loss[2], d_loss, test_val_loss[0], test_val_loss[1])
                print("real_weight\n{}".format(real_weight))
                print("real_labels\n{}".format(real_labels))
                print("layer1_out:train\n{}".format(np.round(fake_weight, decimals=2)))  # 訓練データ時
                if ite % cf.Save_train_step == 0:
                    if irisflag:
                        print("labels:{}".format(np.argmax(y_train, axis=1)))
                        show_result(input=X_train, onehot_labels=y_train,
                                    layer1_out=np.round(g.predict(X_train, verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_train, verbose=0)[0], decimals=2), testflag=False)
                        show_result(input=X_test, onehot_labels=y_test,
                                    layer1_out=np.round(g.predict(X_test, verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_test, verbose=0)[0], decimals=2), testflag=True)
                    else:
                        show_result(input=X_train[:100], onehot_labels=y_train[:100],
                                    layer1_out=np.round(g.predict(X_train[:100], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_train[:100], verbose=0)[0], decimals=2),
                                    testflag=False)
                        show_result(input=X_test[:100], onehot_labels=y_test[:100],
                                    layer1_out=np.round(g.predict(X_test[:100], verbose=0)[1], decimals=2), ite=ite,
                                    classify=np.round(g.predict(X_test[:100], verbose=0)[0], decimals=2), testflag=True)
                        # for i in [1]:
                        # weights 結果をplot
                        # w1 = classify.layers[i].get_weights()[0]
                        # print("w1\n{}".format(w1))
                        # plt.imshow(w1, cmap='coolwarm', interpolation='nearest')
                        # plt.colorbar()
                        # plt.figure()
                        # plt.plot((w1 ** 2).mean(axis=1), 'o-')
                        # plt.show()
            else:
                con += "Ite:{}, catego:{} g:{}, d: {:.6f}".format(ite, g_loss[1], g_loss[2], d_loss)
            sys.stdout.write("\r" + con)

            if ite % cf.Save_train_step == 0 or ite == 1:
                print()
                f.write("{},{},{}{}".format(ite, g_loss, d_loss, os.linesep))
                # save weights
                d.save_weights(cf.Save_d_path)
                g.save_weights(cf.Save_g_path)
                c.save_weights(cf.Save_c_path)
                classify.save_weights(cf.Save_classify_path)

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
        d.save_weights(cf.Save_d_path)
        g.save_weights(cf.Save_g_path)
        c.save_weights(cf.Save_c_path)
        classify.save_weights(cf.Save_classify_path)
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path, cf.Save_classify_path)
        print("maxAcc:{}".format(max_score * 100))
        ### add for TensorBoard
        KTF.set_session(old_session)
        ###


def show_result(input, onehot_labels, layer1_out, ite, classify, testflag=False, showflag=False):
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
            if irisflag:
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
                     testflag, showflag=showflag)
    # visualize([_outs[1] for _outs in layer1_outs], [_outs[2] for _outs in layer1_outs], labels_scalar, ite,
    # testflag, showflag=False)


class Main_test():
    def __init__(self):
        pass

    def test(self):
        ite = 0
        if irisflag:
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = iris_data()
        else:
            X_train, X_test, y_train, y_test, train_num_per_step, data_inds, max_ite = mnist_data()

        g, d, c, classify = weightGAN_Model(input_size=input_size, wSize=wSize, output_size=output_size,
                                            use_mbd=use_mbd)
        g.load_weights(cf.Save_g_path)
        d.load_weights(cf.Save_d_path)
        c.load_weights(cf.Save_c_path)
        classify.load_weights(cf.Save_classify_path)

        # t = np.array([0] * len(X_train))
        # g_loss = c.train_on_batch([X_train, y_train], [y_train, t])  # 生成器を学習
        test_val_loss = classify.evaluate(X_test, y_test)
        train_val_loss = classify.evaluate(X_train, y_train)
        # test_val_loss =  g.evaluate([X_train, y_train], [y_train, t]) # c.evaluate(X_test, y_test)
        # train_val_loss = g.evaluate(X_train, y_train)
        im_train = show_result(input=X_train, onehot_labels=y_train,
                               layer1_out=g.predict(X_train, verbose=0)[1],
                               ite=cf.Iteration, classify=np.round(g.predict(X_train, verbose=0)[0]), testflag=False,
                               showflag=True)
        im_test = show_result(input=X_test, onehot_labels=y_test,
                              layer1_out=g.predict(X_test, verbose=0)[1],
                              ite=cf.Iteration, classify=np.round(g.predict(X_test, verbose=0)[0]), testflag=True, showflag=True)
        print("Ite:{}, train: loss :{:.6f} acc:{:.6f} test_val: loss:{:.6f} acc:{:.6f}"
              .format(ite, train_val_loss[0], train_val_loss[1], test_val_loss[0], test_val_loss[1]))
        weights = classify.get_weights()
        print(weights)
        for i in range(len(weights)):
            print(np.shape(weights[i]))
        classify.summary()
        im_architecture = mydraw(weights, test_val_loss[1])

        im_h_resize = hconcat_resize_min([np.array(im_train), np.array(im_test)])
        im_h_resize = hconcat_resize_min([im_h_resize, np.array(im_architecture)])
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
    parser.add_argument('--mnist', dest='mnist', action='store_true')
    parser.add_argument('--cifar10', dest='cifar10', action='store_true')
    parser.add_argument('--iris', dest='iris', action='store_true')
    parser.add_argument('--krkopt', dest='krkopt', action='store_true')
    parser.add_argument('--use_mbd', dest='use_mbd', action='store_true')
    parser.add_argument('--wSize', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    use_mbd = False
    if args.wSize:
        wSize = args.wSize
    if args.use_mbd:
        use_mbd = True
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
        irisflag = True
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
        krkoptflag = True
    if args.train:
        main = Main_train()
        main.train(use_mbd=use_mbd)
    if args.test:
        main = Main_test()
        main.test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")

#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import tensorflow as tf
from keras import backend as K
from keras import metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import argparse
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# import config_mnist as cf

# from _model_mnist import *


from keras.utils import np_utils

Height, Width = 28, 28
Channel = 1


class Main_train():
    def __init__(self):
        pass

    def train(self, irisflag=False):
        # 性能評価用パラメータ
        max_score = 0.

        ## Load network model
        Size = 4
        wSize = 10
        inputs_z = Input(shape=(Size,), name='Z')  # 入力を取得
        g_dense1 = Dense(wSize, activation='relu', name='g_dense1')(inputs_z)
        x = Dense(3, activation='softmax', name='x_out')(g_dense1)
        print("g_dense1:{}".format(g_dense1))

        # 識別機を学習
        d_dense1 = Dense(10, activation='relu', name='d_dense1')
        d_out = Dense(1, activation='sigmoid', name='d_out')

        # true画像の入力の時(Gの出力を外部に吐き出し、それを入力すると勾配計算がされない)
        inputs_w = Input(shape=(wSize,), name='weight')  # 入力重みを取得
        d_out_true = d_dense1(inputs_w)
        d_out_true = d_out(d_out_true)

        d_out_fake = d_dense1(g_dense1)
        d_out_fake = d_out(d_out_fake)

        g = Model(inputs=[inputs_z], outputs=[x, g_dense1], name='G')
        d = Model(inputs=[inputs_w], outputs=[d_out_true], name='D')
        c = Model(inputs=[inputs_z], outputs=[x, d_out_fake], name='C')
        classify = Model(inputs=[inputs_z], outputs=[x], name='classify')
        classify.compile(loss='categorical_crossentropy',
                         optimizer="adam",
                         metrics=[metrics.categorical_accuracy])
        #  生成器の学習時は識別機は固定
        for layer in d.layers:
            layer.trainable = False
        c_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        c.compile(optimizer=c_opt,
                  loss={'x_out': 'categorical_crossentropy', 'd_out': 'mse'},
                  loss_weights={'x_out': 0.1, 'd_out': 0.9})
        for layer in d.layers:
            layer.trainable = True
        d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        d.compile(optimizer=d_opt, loss='mean_squared_error')
        g.summary()
        d.summary()
        c.summary()
        classify.summary()

        ## Prepare Training data　前処理
        # dl_train = DataLoader(phase='Train', shuffle=True)
        if irisflag:
            # print("data\n{}".format(iris.data))
            # print("target\n{}".format(iris.target))
            X_train, X_test, y_train, y_test \
                = train_test_split(iris.data, iris.target, test_size=0.2, train_size=0.8, shuffle=True)
            # X_train = iris.data[:150]
            y_train = np_utils.to_categorical(y_train, 3)
            # X_test = iris.data[:150]
            y_test = np_utils.to_categorical(y_test, 3)
            train_num = X_train.shape[0]
            train_num_per_step = train_num // cf.Minibatch
            data_inds = np.arange(train_num)
            max_ite = cf.Minibatch * train_num_per_step
        else:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            if X_train.ndim == 3:
                X_train = X_train[:, :, :, None]
            train_num = X_train.shape[0]
            train_num_per_step = train_num // cf.Minibatch
            data_inds = np.arange(train_num)
            max_ite = cf.Minibatch * train_num_per_step
            # クラス分類モデル用に追加
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)
            X_test = (X_test.astype(np.float32) - 127.5) / 127.5
            if X_test.ndim == 3:
                X_test = X_test[:, :, :, None]

        ## Start Train
        print('-- Training Start!!')
        if cf.Save_train_combine is None:
            print("generated image will not be stored")
        elif cf.Save_train_combine is True:
            print("generated image write combined >>", cf.Save_train_img_dir)
        elif cf.Save_train_combine is False:
            print("generated image write separately >>", cf.Save_train_img_dir)

        if irisflag:
            fname = os.path.join(cf.Save_dir, 'loss_iris.txt')
        else:
            fname = os.path.join(cf.Save_dir, 'loss.txt')
        f = open(fname, 'w')
        f.write("Iteration,G_loss,D_loss{}".format(os.linesep))
        print("X_train:{} X_test:{}".format(X_train.shape, X_test.shape))
        print("y_train:{} y_test:{}".format(y_train.shape, y_test.shape))
        # exit()
        for ite in range(cf.Iteration):
            ite += 1
            # Discremenator training
            # y = dl_train.get_minibatch(shuffle=True)
            # print("train_num_per_step:{} minibatch:{}".format(train_num_per_step, cf.Minibatch))
            train_ind = ite % (train_num_per_step - 1)
            if ite % (train_num_per_step + 1) == max_ite:
                np.random.shuffle(data_inds)

            _inds = data_inds[train_ind * cf.Minibatch: (train_ind + 1) * cf.Minibatch]
            # x_fake = X_train[_inds]
            real_weight = np.zeros((cf.Minibatch, wSize))
            for i in range(cf.Minibatch):
                if y_train[_inds][i][0] == 1:
                    real_weight[i][0] = 1
                elif y_train[_inds][i][1] == 1:
                    real_weight[i][1] = 1
                elif y_train[_inds][i][2] == 1:
                    real_weight[i][2] = 1
            # real_weight = real_weight * g.predict(X_train[_inds], verbose=0)[1]
            fake_weight = g.predict(X_train[_inds], verbose=0)[1]
            # print("X_train:{}".format(np.shape(X_train[_inds])))
            # print("y_train:{}".format(np.shape(y_train[_inds])))
            # print("g_out:{}".format(np.shape(g.predict(X_train[_inds])[0])))
            # print("real_weight:{}".format(np.shape(real_weight)))
            # print("fake_weight:{}".format(np.shape(fake_weight)))
            # print("t:{}".format(np.shape(t)))
            # print("d_out:{}".format(np.shape(d.predict(fake_weight, verbose=0))))

            t = np.array([1] * cf.Minibatch + [0] * cf.Minibatch)
            concated_weight = np.concatenate((fake_weight, real_weight))
            if ite % 1 == 0:
                d_loss = d.train_on_batch(concated_weight, t)  # これで重み更新までされる
            else:
                d_loss = d.test_on_batch(concated_weight, t)  # これで重み更新までされる
            t = np.array([[0] for _ in range(cf.Minibatch)])  # [0] * cf.Minibatch
            g_loss = c.train_on_batch([X_train[_inds]], [y_train[_inds], t])  # 生成器を学習
            # d_loss = classify.train_on_batch(X_train[_inds], y_train[_inds])[1]  # これで重み更新までされる
            con = '|'
            if ite % cf.Save_train_step != 0:
                for i in range(ite % cf.Save_train_step):
                    con += '>'
                for i in range(cf.Save_train_step - ite % cf.Save_train_step):
                    con += ' '
            else:
                for i in range(cf.Save_train_step):
                    con += '>'
            con += '| '
            if ite % 100 == 0:
                # test_val_loss = classify.evaluate(X_test, y_test)
                test_val_loss = classify.evaluate(X_train, y_train)  # 低層CNNの性能測定
                max_score = max(max_score, 1. - test_val_loss[1])
                con += "Ite:{}, catego: {} g: {}, d: {:.6f}, test_val: loss:{:.6f} acc:{:.6f}".format(ite, g_loss[1], g_loss[2], d_loss, test_val_loss[0], test_val_loss[1])
                # print("fake_weight\n{}".format(fake_weight))
                # print("real_weight\n{}".format(real_weight))
                print("real_weight\n{}".format(real_weight[:10]))
                print("layer1_out\n{}".format(np.round(fake_weight[:10], decimals=2)))
                # for i in range(10):
                # print(classify.layers[i].get_weights())
                # print(classify.layers[i].get_weights()[0])
                if ite % 10000 == 0:
                    print("layer1_out\n{}".format(g.predict(X_test, verbose=0)[1]))
                    for i in [1]:
                        # weights 結果をplot
                        w1 = classify.layers[i].get_weights()[0]
                        print("w1\n{}".format(w1))
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
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path)
        print("maxAcc:{}".format(max_score * 100))


class Main_test():
    def __init__(self):
        pass

    def test(self):
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        if args.mnist:
            Height = 28
            Width = 28
            Channel = 1
            from keras.datasets import mnist
            import config_mnist as cf
            from _model_mnist import *

            np.random.seed(cf.Random_seed)
            main = Main_train()
            main.train(irisflag=True)
        elif args.cifar10:
            Height = 32
            Width = 32
            Channel = 3
            import config_cifar10 as cf
            from keras.datasets import cifar10 as mnist
            from _model_cifar10 import *

            np.random.seed(cf.Random_seed)
            main = Main_train()
            main.train(irisflag=True)
        elif args.iris:
            Height = 1
            Width = 3
            Channel = 1
            import config_mnist as cf
            from _model_iris import *

            np.random.seed(cf.Random_seed)
            from sklearn.datasets import load_iris

            iris = load_iris()
            main = Main_train()
            main.train(irisflag=True)
    if args.test:
        main = Main_test()
        main.test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")

from __future__ import division
import sys, os, pickle

import numpy as np
import numpy.random as rd

from scipy.misc import imread

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

image_size = 28
depth = 255


def unpickle(filename):
    with open(filename, 'rb') as fo:
        _dict = pickle.load(fo)
    return _dict


def to_pickle(filename, obj):
    with open(filename, 'wb') as f:
        # pickle.dump(obj, f, -1)
        pickle.Pickler(f, protocol=2).dump(obj)


def count_empty_file(folder):
    cnt = 0
    for file in os.listdir(folder):
        if os.stat(os.path.join(folder, file)).st_size == 0:
            cnt += 1
    return cnt


def get_data():
    label_conv = {a: i for a, i in zip('ABCDEFGHIJ', range(10))}

    assert os.path.exists('notMNIST_large')
    assert os.path.exists('notMNIST_small')
    print("load_start")
    for root_dir in ['notMNIST_small', 'notMNIST_large']:
        folders = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
                   if os.path.isdir(os.path.join(root_dir, d))]

        file_cnt = 0
        for folder in folders:
            label_name = os.path.basename(folder)
            file_list = os.listdir(folder)
            file_cnt += len(file_list) - count_empty_file(folder)

        dataset = np.ndarray(shape=(file_cnt, image_size * image_size), dtype=np.float32)
        labels = np.ndarray(shape=(file_cnt), dtype=np.int)

        last_num = 0  # 前の文字の最終インデックス

        for folder in folders:

            file_list = os.listdir(folder)
            file_cnt = len(file_list) - count_empty_file(folder)

            label_name = os.path.basename(folder)
            labels[last_num:(last_num + file_cnt)] = label_conv[label_name]

            skip = 0
            for i, file in enumerate(file_list):

                # ファイルサイズが0のものはスキップ
                if os.stat(os.path.join(folder, file)).st_size == 0:
                    skip += 1
                    continue
                try:
                    data = imread(os.path.join(folder, file))
                    data = data.astype(np.float32)
                    data /= depth  # 0-1のデータに変換
                    dataset[last_num + i - skip, :] = data.flatten()
                except:
                    skip += 1
                    print('error {}'.format(file))
                    continue
            last_num += i - skip

        notmnist = {}
        notmnist['data'] = dataset
        notmnist['target'] = labels
        to_pickle('{}.pkl'.format(root_dir), notmnist)

    notmnist = unpickle('notMNIST_large.pkl')  # 同じフォルダにnotMNIST_large.pkl が入っているとする。
    notmnist_data = notmnist['data']
    notmnist_target = notmnist['target']

    notmnist_data = notmnist_data.astype(np.float32)
    notmnist_target = notmnist_target.astype(np.int32)
    notmnist_data /= 255.  # 0-1のデータに変換

    return train_test_split(notmnist_data, notmnist_target)
    # 学習用データを 75%、検証用データを残りの個数と設定
    # x_train, x_test, y_train, y_test = train_test_split(notmnist_data, notmnist_target)
    # [draw_digit([[notmnist_data[idx], notmnist_target[idx]] for idx in rd.randint(len(notmnist_data), size=10)]) for i in range(10)]

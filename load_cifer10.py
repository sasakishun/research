from __future__ import unicode_literals, print_function
import numpy as np


def unpickle(f):
    import _pickle
    fo = open(f, 'rb')
    d = _pickle.load(fo, encoding='latin1')
    fo.close()
    return d


def load_dataset():
    label_names = unpickle("cifar10/batches.meta")["label_names"]

    d = unpickle("cifar10/data_batch_1")
    d2 = unpickle("cifar10/data_batch_2")
    d3 = unpickle("cifar10/data_batch_3")
    d4 = unpickle("cifar10/data_batch_4")
    d5 = unpickle("cifar10/data_batch_5")

    data = d["data"]
    labels = np.array(d["labels"])
    data = np.r_[data, d2["data"]]
    data = np.r_[data, d3["data"]]
    data = np.r_[data, d4["data"]]
    data = np.r_[data, d5["data"]]
    labels = np.r_[labels, np.array(d2["labels"])]
    labels = np.r_[labels, np.array(d3["labels"])]
    labels = np.r_[labels, np.array(d4["labels"])]
    labels = np.r_[labels, np.array(d5["labels"])]

    tester = unpickle("cifar10/test_batch")
    test_data = tester["data"]
    test_labels = np.array(tester["labels"])

    return data, labels, test_data, test_labels, label_names

import numpy as np
import os
import chainer
from chainer import datasets

train, test = datasets.get_cifar100()
cifar_train_data = [train[j][0].reshape(32*32*3) for j in range(len(train))]
print("train_data_complete")
cifar_train_label_ = [np.eye(100)[int(train[j][1])] for j in range(len(train))]
print("test_data_complete")
cifar_test_data = [test[j][0].reshape(32*32*3) for j in range(len(test))]
print("train_label_complete")
cifar_test_label_ = [np.eye(100)[int(test[j][1])] for j in range(len(test))]
print("test_label_complete")

np.save("cifar-100_train_data", cifar_train_data)
np.save("cifar-100_test_data", cifar_test_data)
np.save("cifar-100_train_label", cifar_train_label_)
np.save("cifar-100_test_label", cifar_test_label_)
print(cifar_test_label_[0])


"""
def unpickle(f):
    import _pickle
    fo = open(f, 'rb')
    d = _pickle.load(fo, encoding='latin1')
    fo.close()
    return d


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(folder):
    tr_data = np.empty((0, 32 * 32 * 3))
    tr_labels = np.empty(1)
    for i in range(1, 6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            tr_data = data_dict['data']
            tr_labels = data_dict['labels']
        else:
            tr_data = np.vstack((tr_data, data_dict['data']))
            tr_labels = np.hstack((tr_labels, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    te_data = data_dict['data']
    te_labels = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    label_names = bm['label_names']
    return tr_data, tr_labels, te_data, te_labels, label_names


def get_cifar100(folder):
    train_fname = os.path.join(folder, 'train')
    test_fname = os.path.join(folder, 'test')
    data_dict = unpickle(train_fname)
    train_data = data_dict['data']
    train_fine_labels = data_dict['fine_labels']
    train_coarse_labels = data_dict['coarse_labels']

    data_dict = unpickle(test_fname)
    test_data = data_dict['data']
    test_fine_labels = data_dict['fine_labels']
    test_coarse_labels = data_dict['coarse_labels']

    bm = unpickle(os.path.join(folder, 'meta'))
    clabel_names = bm['coarse_label_names']
    flabel_names = bm['fine_label_names']

    return train_data, np.array(train_coarse_labels), np.array(train_fine_labels), test_data, np.array(
        test_coarse_labels), np.array(test_fine_labels), clabel_names, flabel_names


if __name__ == '__main__':
    datapath2 = "./data_cifar-100/cifar-100-python"

    tr_data100, tr_clabels100, tr_flabels100, te_data100, te_clabels100, te_flabels100, clabel_names100, flabel_names100 \
        = get_cifar100(datapath2)

    img0 = tr_data100[0]
    img0 = img0.reshape((3,32,32))
    img1 = np.rollaxis(img0, 0, 3)
    from skimage import io
    io.imshow(img1)
    io.show()
"""
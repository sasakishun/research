import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_iris
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# make Data_set_for_numpy
"""
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
data = mnist.train.images
labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
validation_data = mnist.validation.images
validation_labels = mnist.validation.labels

np.save("MNIST_train_data", data)
np.save("MNIST_train_labels", labels)
np.save("MNIST_test_data", test_data)
np.save("MNIST_test_labels", test_labels)
np.save("MNIST_valid_data", validation_data)
np.save("MNIST_valid_labels", validation_labels)

# load Data_set
data = np.load("MNIST_train_data.npy")
labels = np.load("MNIST_train_labels.npy")
test_data = np.load("MNIST_test_data.npy")
test_labels = np.load("MNIST_test_labels.npy")
"""

"""
iris = load_iris()
train_data_num = [i for i in range(len(iris.target))]
print(train_data_num)
np.random.shuffle(train_data_num)
data = [[0. for i in range(len(iris.data[0]))] for j in range(len(iris.data))]
label = [[0 for i in range(3)] for j in range(len(iris.data))]
print(data)
print(label)
for i in range(len(train_data_num)):
    data[i] = iris.data[train_data_num[i]]
for i in range(len(train_data_num)):
    label[i] = np.eye(3)[int(iris.target[train_data_num[i]])]

print(data)
print(label)
np.save("iris_train_data", data[0:100])
np.save("iris_train_labels", label[0:100])
np.save("iris_test_data", data[100:150])
np.save("iris_test_labels", label[100:150])
"""

"""
data = np.load("cifar10_train_data.npy")
labels = np.load("cifar10_train_labels.npy")
test_data = np.load("cifar10_test_data.npy")
test_labels = np.load("cifar10_test_labels.npy")

cifar_data = [[0.0 for j in range(len(data[i]))] for i in range(len(data))]
cifar_test_data = [[0.0 for j in range(len(test_data[i]))] for i in range(len(test_data))]

print(data[0])
for i in range(len(data)):
    print(i)
    max = data[i].max()
    for j in range(len(data[i])):
        cifar_data[i][j] = float(data[i][j]/max)
for i in range(len(test_data)):
    print(i)
    max = test_data[i].max()
    for j in range(len(test_data[i])):
        cifar_test_data[i][j] = float(test_data[i][j]/max)

np.save("cifar-10_train_data_normalized", cifar_data)
np.save("cifar-10_train_labels", labels)
np.save("cifar-10_test_data_normalized", cifar_test_data)
np.save("cifar-10_test_labels", test_labels)
"""

"""
data = mnist.train.images
labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

np.save("fashion_train_data", data)
np.save("fashion_train_labels", labels)
np.save("fashion_test_data", test_data)
np.save("fashion_test_labels", test_labels)

print(data[0])
print(labels[0])
print(data.shape)
print(labels.shape)
"""
"""
mnist = input_data.read_data_sets("./FASHION/", one_hot=True)
data = mnist.train.images
labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
validation_data = mnist.validation.images
validation_labels = mnist.validation.labels
np.save("fashion_train_data", data)
np.save("fashion_train_labels", labels)
np.save("fashion_test_data", test_data)
np.save("fashion_test_labels", test_labels)
np.save("fashion_valid_data", validation_data)
np.save("fashion_valid_labels", validation_labels)

print(data[0])
print(data.shape)
# 画像の読み込み
im = np.reshape(data[1], [28, 28])

# 貼り付け
plt.imshow(im)
# 表示
plt.show()
"""
data = np.load("cifar-100_train_data.npy")
labels = np.load("cifar-100_train_label.npy")
test_data = np.load("cifar-100_test_data.npy")
test_labels = np.load("cifar-100_test_label.npy")
print(test_labels[0])
print(labels[0])
for i in range(100):
    print(data[i]-test_data[i])

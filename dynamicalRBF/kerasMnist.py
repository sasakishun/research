'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import rbflayer

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

import activations
batch_size = 32
num_classes = 10
epochs = 1000
rbfNodes = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# data = x_train# [:, :-1]
# data = data.reshape(60000, 784)
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# data = [data[i] for i in range(np.shape(data)[0])]
# print("mnist  data:{}".format(np.shape(data)))
"""
model.add(rbflayer.RBFLayer(output_dim=rbfNodes,# セントロイドの数
                            initializer=rbflayer.InitCentersRandom(data),
                            betas=1.0,
                            input_shape=(784, )))
"""
activation = "relu"# activations.swish
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation,
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(rbflayer.RBFLayer(output_dim=rbfNodes,# セントロイドの数
                            initializer=None,
                            betas=1.0))
# model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu", input_shape=(784, )))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Dense(num_classes, activation=activation))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

### add for TensorBoard
tb_cb = keras.callbacks.TensorBoard(log_dir="tflog", histogram_freq=1)
cbks = [tb_cb]
###

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
scoreTrain = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

### add for TensorBoard
KTF.set_session(old_session)
###

### save result -> log.txt
with open("log.txt", mode='w') as f:
    f.write("batchsize:{} epoch:{} rbfNode:{} traAcc:{} valAcc:{}".format(batch_size, epochs, rbfNodes, scoreTrain[1], score[1]))
###
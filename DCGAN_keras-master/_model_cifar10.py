import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, \
    UpSampling2D, LeakyReLU, Activation, Conv2DTranspose

from keras.initializers import RandomNormal as RN, Constant


def G_model(Height, Width, channel=3):
    inputs_z = Input((Height, Width, channel), name='Z')  # 入力画像を取得
    x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), name='g_conv1',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs_z)
    """
    x = InstanceNormalization()(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), name='g_conv2',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), name='g_conv3',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv3_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (5, 5), padding='same', strides=(2, 2), name='g_conv4',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    """
    x = Conv2D(256, (5, 5), padding='same', strides=(8, 8), name='g_conv4',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv4_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    model = Model(inputs=[inputs_z], outputs=[x], name='G')
    return model, [Height // 16, Width // 16, 256]  # [2, 2, 256]


def D_model(Height, Width, channel=3):
    inputs = Input((Height, Width, channel))
    x = Conv2D(32, (1, 1), padding='same', strides=(1, 1), name='d_conv1',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), name='d_conv2',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (1, 1), padding='same', strides=(1, 1), name='d_conv3',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (1, 1), padding='same', strides=(1, 1), name='d_conv4',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    # x = Dense(2048, activation='relu', name='d_dense1',
    #    kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    x = Dense(1, activation='sigmoid', name='d_out',
              kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name='D')
    return model


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model


def conv_x(Height, Width, channel=3):
    inputs_z = Input((Height, Width, channel), name='Z')  # 入力画像を取得
    x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), name='g_conv1',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs_z)
    x = InstanceNormalization()(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), name='g_conv2',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), name='g_conv3',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='d_conv3_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (5, 5), padding='same', strides=(2, 2), name='g_conv4',
               kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv4_bn')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    model = Model(inputs=[inputs_z], outputs=[x], name='G')
    return model


def classifying(Height, Width, channel=3, num_classes=10):
    inputs_x = Input((Height, Width, channel), name='classifying_X')  # generator出力を取得
    x = Flatten()(inputs_x)
    x = Dense(num_classes, activation='sigmoid', name='d_out',
              kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs_x, outputs=x, name='D')

    return model


from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class InstanceNormalization(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        if K.backend() != "tensorflow":
            raise ValueError("only tf")

        if len(input_shape) == 2:
            mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
            return (inputs - mean) / K.sqrt(var + K.epsilon())
        elif len(input_shape) == 4 and K.image_data_format() == "channels_last":
            mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            return (inputs - mean) / K.sqrt(var + K.epsilon())
        else:
            raise ValueError("Not valid")

    def compute_output_shape(self, input_shape):
        return input_shape

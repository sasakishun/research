import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, \
    UpSampling2D, concatenate
from keras.applications.vgg16 import VGG16


def G_model(Size=4, wSize=4*5):
    inputs_z = Input(shape=(Size,), name='Z')  # 入力を取得
    layer1 = Dense(5, activation='relu', name='g_dense1')
    x = layer1(inputs_z)
    x = Dense(3, activation='softmax', name='g_dense2')(x)
    model_classify = Model(inputs=[inputs_z], outputs=[x], name='classify')
    # 識別機を学習
    inputs_weight = Input(shape=(wSize,), name='weight')  # 入力重みを取得
    concat = concatenate()([Flatten(layer1.kernel), Flatten(inputs_weight)])
    concat = Dense(100, activation='relu', name='d_dense1')(concat)
    concat = Dense(100, activation='relu', name='d_dense2')(concat)
    model = Model(inputs=[inputs_z], outputs=[x, concat], name='G')
    return model_classify

def D_model(Height, Width, channel=3):
    inputs_x = Input((Height, Width, channel), name='X')  # generator出力を取得
    x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv1')(inputs_x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='d_dense1')(x)
    x = Dense(1, activation='sigmoid', name='d_out')(x)
    model = Model(inputs=[inputs_x], outputs=[x], name='D')
    return model


def conv_x(Height, Width, channel=3):
    inputs_x = Input((Height, Width, channel), name='conv_X')  # generator出力を取得
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='cnn_conv1')(inputs_x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='cnn_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    model = Model(inputs=[inputs_x], outputs=[x], name='conv_x')
    return model


def classifying(Height, Width, channel=3, num_classes=10):
    inputs_x = Input((Height, Width, channel), name='classifying_X')  # generator出力を取得
    x = Flatten()(inputs_x)
    x = Dense(128, activation='relu', name='cnn_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='cnn_out')(x)
    model = Model(inputs=[inputs_x], outputs=[x], name='classifying')
    return model


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

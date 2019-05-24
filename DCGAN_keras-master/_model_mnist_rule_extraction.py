import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, \
    UpSampling2D, concatenate
from keras.applications.vgg16 import VGG16


def G_model(Height, Width, channel=3):
    """
    inputs_z = Input((100,), name='Z')
    in_h = int(Height / 4)
    in_w = int(Width / 4)
    x = Dense(in_h * in_w * 128, activation='tanh', name='g_dense1')(inputs_z)
    x = BatchNormalization()(x)
    x = Reshape((in_h, in_w, 128), input_shape=(128 * in_h * in_w,))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='g_conv1')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(channel, (5, 5), padding='same', activation='tanh', name='g_out')(x)
    """
    inputs_z = Input((Height, Width, channel), name='Z')  # 入力画像を取得
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='g_conv1')(inputs_z)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='g_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='g_conv3')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='g_conv4')(x)
    x = Conv2D(1, (3, 3), padding='same', activation='relu', name='g_conv5')(x)
    model = Model(inputs=[inputs_z], outputs=[x], name='G')
    return model, [Height, Width, 1]


def D_model(Height, Width, channel=3):
    inputs_x = Input((Height, Width, channel), name='X')  # generator出力を取得
    x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv1')(inputs_x)
    # x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv11')(x)
    # x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv12')(x)
    # x = Conv2D(64, (5, 5), padding='same', activation='tanh', name='d_conv13')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv2')(x)
    # x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv22')(x)
    # x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv23')(x)
    # x = Conv2D(128, (5, 5), padding='same', activation='tanh', name='d_conv24')(x)
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

    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='classify1')(inputs_x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='classify2')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='classify3')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='classify4')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='classify5')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='classify6')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='classify_out')(x)

    model = Model(inputs=[inputs_x], outputs=[x], name='classifying')
    return model


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

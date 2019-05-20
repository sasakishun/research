import keras
import tensorflow as tf
def swish(x):
   beta = 1.5 #1, 1.5 or 2
   # keras.backendモジュールを抜くと勾配計算不可能になる
   # return beta * tf.nn.sigmoid(x)
   return beta * x * keras.backend.sigmoid(x)

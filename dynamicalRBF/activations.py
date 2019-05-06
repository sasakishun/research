import keras

def swish(x):
   beta = 1.5 #1, 1.5 or 2
   return beta * x * keras.backend.sigmoid(x)
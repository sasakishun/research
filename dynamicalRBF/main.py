import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def spiky(x):
    r = x % 1
    if r <= 0.5:
        return r
    else:
        return 0


def d_spiky(x):
    r = x % 1
    if r <= 0.5:
        return 1
    else:
        return 0

np_d_spiky = np.vectorize(d_spiky)
np_d_spiky_32 = lambda x: np_d_spiky(x).astype(np.float32)
def tf_d_spiky(x, name=None):
    with tf.name_scope(name, "d_spiky", [x]) as name:
        y = tf.py_func(np_d_spiky_32,
                       [x],
                       [tf.float32],
                       name=name,
                       stateful=False)
        return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def spikygrad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_spiky(x)
    return grad * n_gr


np_spiky = np.vectorize(spiky)
np_spiky_32 = lambda x: np_spiky(x).astype(np.float32)

def tf_spiky(x, name=None):
    with tf.name_scope(name, "spiky", [x]) as name:
        y = py_func(np_spiky_32,
                    [x],
                    [tf.float32],
                    name=name,
                    grad=spikygrad)  # <-- here's the call to the gradient
        return y[0]


with tf.Session() as sess:
    x = tf.constant([0.2, 0.7, 1.2, 1.7])
    y = tf_spiky(x)
    tf.initialize_all_variables().run()

    print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())

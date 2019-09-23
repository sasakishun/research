from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, regularizers, initializers, constraints

class MyLayer(Layer):
    def __init__(self, output_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.output_dim = output_dim
        # self.activation = activation
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, ),
                                    initializer='zeros')
        super(MyLayer, self).build(input_shape)


    def call(self, x, kernel_mask=None, bias_mask=None):
        if kernel_mask is not None:
            self.kernel = self.kernel * kernel_mask
        if bias_mask is not None:
            self.bias = self.bias * bias_mask
        return self.activation(K.dot(x, self.kernel) + self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = output_units

    def build(self, input_shape):
        # build では変数の宣言と登録を行います。
        # build は最初にレイヤーが実行（callが呼ばれたとき）の1回のみ実行されます。
        # 変数の宣言と登録は、 __init__ で行っても構いません。
        # ただ、 build で行うと入力の shape がわかるため便利というだけです。
        depth = int(input_shape[-1])
        self.kernel = self.add_weight(
            "kernel",
            shape=[depth, self.output_units],
            # 以下は Optional
            dtype=tf.float32,
            initializer=tf.initializers.orthogonal(dtype=tf.float32),
        )
        # self.built = True と同義。最後に実行します。
        super().build(input_shape)

    def call(self, input):
        # 実行時の処理を書きます。
        return tf.matmul(input, self.kernel)

    def call(self, input, mask):
        # 実行時の処理を書きます。
        return tf.matmul(input, tf.mul(self.kernel, mask))

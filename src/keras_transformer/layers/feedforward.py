import tensorflow as tf


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(
        self, d_model, d_ff, dropout=0.1, name="positionwise_feedforward", **kwargs
    ):
        super(PositionwiseFeedForward, self).__init__(name=name, **kwargs)
        self.d1 = tf.keras.layers.Dense(d_ff)
        self.relu = tf.keras.layers.Activation("relu")
        self.d2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.d1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.d2(x)
        return x

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_length=5000,
        name="positional_encoding",
        **kwargs
    ):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)

        position = np.arange(max_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))

        angle_rads = position * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i (even)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+1 (odd)

        pe = angle_rads[np.newaxis, ...]
        self.positional_encoding = tf.cast(pe, dtype=tf.float32)

    def call(self, x):
        x = x + self.positional_encoding[:, : tf.shape(x)[1]]
        return self.dropout(x)

import copy

import tensorflow as tf

from keras_transformer.layers.attention import MultiHeadAttention
from keras_transformer.layers.feedforward import PositionwiseFeedForward
from keras_transformer.layers.normalization import LayerNorm


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, layer: tf.keras.layers.Layer, N: int, name: str = "encoder", **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.layers = [copy.deepcopy(layer) for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def call(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # (batch_size, input_seq_len, d_model)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, num_head, d_ff, dropout=0.1, name="encoder_layer", **kwargs
    ):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.size = d_model

        self.attention = MultiHeadAttention(d_model, num_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

        self.feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask):
        x_norm = self.norm1(x)
        out = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(out)

        x_norm = self.norm2(x)
        out = self.feedforward(x_norm)
        x = x + self.dropout2(out)
        return x  # (batch_size, input_seq_len, d_model)

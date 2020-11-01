import copy

import tensorflow as tf

from keras_transformer.layers.attention import MultiHeadAttention
from keras_transformer.layers.feedforward import PositionwiseFeedForward
from keras_transformer.layers.normalization import LayerNorm


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, layer: tf.keras.layers.Layer, N: int, name: str = "decoder", **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.layers = [copy.deepcopy(layer) for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def call(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, num_head, d_ff, dropout=0.1, name="decoder_layer", **kwargs
    ):
        super(DecoderLayer, self).__init__(name=name, **kwargs)
        self.size = d_model

        self.attention = MultiHeadAttention(d_model, num_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

        self.src_attention = MultiHeadAttention(d_model, num_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, memory, src_mask, tgt_mask):
        x_norm = self.norm1(x)
        out = self.attention(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(out)

        x_norm = self.norm2(x)
        out = self.src_attention(x_norm, memory, memory, src_mask)
        x = x + self.dropout2(out)

        x_norm = self.norm3(x)
        out = self.feedforward(x_norm)
        x = x + self.dropout3(out)
        return x

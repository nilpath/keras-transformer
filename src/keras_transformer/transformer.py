import copy

import tensorflow as tf

from keras_transformer.attention import MultiHeadAttention
from keras_transformer.feedforward import PositionwiseFeedForward
from keras_transformer.normalization import LayerNorm


class EncoderDecoder(tf.keras.layers.Layer):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def call(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layer: tf.keras.layers.Layer, N: int):
        super(Encoder, self).__init__()
        self.layers = [copy.deepcopy(layer) for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def call(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
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
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, layer: tf.keras.layers.Layer, N: int):
        super(Decoder, self).__init__()
        self.layers = [copy.deepcopy(layer) for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def call(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
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

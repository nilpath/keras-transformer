from typing import Tuple

import tensorflow as tf


def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def scaled_dot_product_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: tf.Tensor = None,
    dropout: tf.keras.layers.Dropout = None,
) -> Tuple[tf.Tensor, tf.Tensor]:

    dim_k = tf.shape(query)[-1]

    # NOTE: Do we transpose key (K) because we want the QxK multiplication to be
    # done for corresponding token vector in each matrix Q and K ?
    #
    # [word1, word1, word1]     [word1, word2, word3]
    # [word2, word2, word2] x T [word1, word2, word3]
    # [word3, word3, word3]     [word1, word2, word3]
    #

    scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(
        tf.cast(dim_k, tf.float32)
    )

    if mask is not None:
        scores += mask * -1e9

    p_attn = tf.nn.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    output = tf.matmul(p_attn, value)
    return output, p_attn


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._num_heads = num_heads
        self._d_model = d_model
        assert d_model % num_heads == 0
        self._d_kv = d_model // num_heads

        self._wq = tf.keras.layers.Dense(d_model)  # TODO: Impl our own  dense block?
        self._wk = tf.keras.layers.Dense(d_model)  # TODO: Impl our own  dense block?
        self._wv = tf.keras.layers.Dense(d_model)  # TODO: Impl our own  dense block?
        self._wout = tf.keras.layers.Dense(d_model)  # TODO: Impl our own dense block?

        self._attn = None
        self._dropout = tf.keras.layers.Dropout(dropout)

    def _transform_input(self, x, batch_size):
        # reshape and transpose (batch_size, seq_length, embedding_size) to
        # (batch_size, h, seq_len, embedding_size)
        x = tf.reshape(x, (batch_size, -1, self._num_heads, self._d_kv))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor = None,
    ):
        # q, v, k shapes = (batch_size, seq_length, embedding_size)?
        batch_size = tf.shape(query)[0]

        # Linear layers
        query, key, value = [
            self._transform_input(layer(x), batch_size)
            for layer, x in zip((self._wq, self._wk, self._wv), (query, key, value))
        ]

        # Attention
        x, self._attn = scaled_dot_product_attention(
            query, key, value, mask, dropout=self._dropout
        )

        # Concatenate
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self._d_model))

        # Final linear
        return self._wout(x)

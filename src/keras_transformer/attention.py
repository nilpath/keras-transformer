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
        scores += (mask * -1e9)

    p_attn = tf.nn.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    output = tf.matmul(p_attn, value)
    return output, p_attn

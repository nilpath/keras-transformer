from typing import List, Tuple, Union

import tensorflow as tf


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, num_feat: Union[List[int], Tuple[int]], eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = tf.ones([num_feat])
        self.b_2 = tf.zeros([num_feat])
        self.eps = eps

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

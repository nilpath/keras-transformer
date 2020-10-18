import tensorflow as tf


def create_padding_mask(seq: tf.Tensor, pad_val: int = 0) -> tf.Tensor:
    seq = tf.cast(tf.math.equal(seq, pad_val), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

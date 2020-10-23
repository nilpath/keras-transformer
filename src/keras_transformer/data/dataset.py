import numpy as np
import tensorflow as tf


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches * batch_size, vec_length)
    data = np.random.randint(vec_min, vec_max, size=size)

    return tf.data.Dataset.from_tensor_slices((data, data)).padded_batch(
        batch_size=batch_size
    )


def create_text_dataset(
    source, target, src_encoder, tgt_encoder, batch_size=64, seq_length=128, buffer_size=20000
):

    def enc(source: tf.Tensor, target: tf.Tensor):
        src = src_encoder.encode("<SOS>") + src_encoder.encode(source.numpy()) + src_encoder.encode("<EOS>")
        tgt = tgt_encoder.encode("<SOS>") + tgt_encoder.encode(target.numpy()) + tgt_encoder.encode("<EOS>")
        return src, tgt

    def tf_enc(source: tf.Tensor, target: tf.Tensor):
        src, tgt = tf.py_function(enc, [source, target], [tf.int64, tf.int64])
        src.set_shape([None])
        tgt.set_shape([None])
        return src, tgt

    def max_length_filter(source: tf.Tensor, target: tf.Tensor, max_length=seq_length):
        return tf.logical_and(
            tf.size(source) <= max_length,
            tf.size(target) <= max_length,
        )

    dataset = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .map(tf_enc)
        .filter(max_length_filter)
        .cache()
        .shuffle(buffer_size)
        .padded_batch(batch_size)
    )

    return dataset

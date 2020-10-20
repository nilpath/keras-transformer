import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches * batch_size, vec_length)
    data = np.random.randint(vec_min, vec_max, size=size)

    return tf.data.Dataset.from_tensor_slices((data, data)).padded_batch(
        batch_size=batch_size
    )


def create_text_dataset(source, target, prefix=None, postfix=None):
    reserved_tokens = []
    if prefix is not None:
        source = [prefix + c for c in source]
        target = [prefix + c for c in target]
        reserved_tokens.append(prefix)

    if postfix is not None:
        source = [c + postfix for c in source]
        target = [c + postfix for c in target]
        reserved_tokens.append(postfix)

    tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        target, target_vocab_size=2**13, reserved_tokens=reserved_tokens
    )

    src_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        source, target_vocab_size=2**13, reserved_tokens=reserved_tokens
    )

    def enc(source: tf.Tensor, target: tf.Tensor):
        src = src_encoder.encode(source.numpy())
        tgt = tgt_encoder.encode(target.numpy())
        return src, tgt

    def tf_enc(source: tf.Tensor, target: tf.Tensor):
        src, tgt = tf.py_function(enc, [source, target], [tf.int64, tf.int64])
        src.set_shape([None])
        tgt.set_shape([None])
        return src, tgt

    def max_length_filter(source: tf.Tensor, target: tf.Tensor, max_length=40):
        return tf.logical_and(
            tf.size(source) <= max_length,
            tf.size(target) <= max_length,
        )

    dataset = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .map(tf_enc)
        .filter(max_length_filter)
        .cache()
        .shuffle(20000)
        .padded_batch(64)
    )

    return dataset, src_encoder, tgt_encoder

import numpy as np
import tensorflow as tf


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches * batch_size, vec_length)
    data = np.random.randint(vec_min, vec_max, size=size)

    return tf.data.Dataset.from_tensor_slices((data, data)).padded_batch(
        batch_size=batch_size
    )


def create_text_dataset(
    source,
    target,
    src_encoder,
    tgt_encoder,
    batch_size=64,
    shuffle=True,
    buffer_size=20000,
):
    def enc(source: tf.string, target: tf.string):
        src = (
            src_encoder.encode("<SOS>")
            + src_encoder.encode(source.numpy())
            + src_encoder.encode("<EOS>")
        )
        tgt = (
            tgt_encoder.encode("<SOS>")
            + tgt_encoder.encode(target.numpy())
            + tgt_encoder.encode("<EOS>")
        )
        return src, tgt

    def tf_enc(source: tf.Tensor, target: tf.Tensor):
        src, tgt = tf.py_function(enc, [source, target], [tf.int64, tf.int64])
        src.set_shape([None])
        tgt.set_shape([None])
        return src, tgt

    dataset = tf.data.Dataset.from_tensor_slices((source, target))
    dataset = dataset.map(tf_enc)
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    def element_length_func(x, y):
        return tf.shape(x)[0]

    bucket_bounderies = [512, 256, 128, 64, 32, 16]
    bucket_batch_sizes = [batch_size] * (len(bucket_bounderies) + 1)

    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            bucket_bounderies,
            bucket_batch_sizes,
            drop_remainder=True,
            pad_to_bucket_boundary=False,
        )
    )

    return dataset

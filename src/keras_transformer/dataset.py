import numpy as np
import tensorflow as tf


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches * batch_size, vec_length)
    # Should probably be one hot? (batch, seq_length, one_host vec max?)
    label_size = (nbatches * batch_size, vec_length, vec_max)
    data = np.random.randint(vec_min, vec_max, size=size)
    labels = np.random.randint(vec_min, vec_max, size=label_size)

    # def with_labels(src, tgt):
    #     tgt_in = tgt[:, :-1]
    #     tgt_out = tgt[:, 1:]
    #     return (src, tgt_in), tgt_out

    # return (
    #     tf.data.Dataset.from_tensor_slices((data, data))
    #     .map(with_labels)
    #     .batch(batch_size=batch_size)
    # )

    return tf.data.Dataset.from_tensor_slices(((data, data), labels)).batch(
        batch_size=batch_size
    )

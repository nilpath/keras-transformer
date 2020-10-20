import numpy as np
import tensorflow as tf


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches * batch_size, vec_length)
    data = np.random.randint(vec_min, vec_max, size=size)

    return tf.data.Dataset.from_tensor_slices((data, data)).padded_batch(
        batch_size=batch_size
    )

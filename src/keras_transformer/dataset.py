import numpy as np
import tensorflow as tf

from keras_transformer.attention import create_look_ahead_mask


def synthetic_data(nbatches, batch_size, vec_length, vec_min=1, vec_max=11):
    size = (nbatches*batch_size, vec_length)
    data = np.random.randint(vec_min, vec_max, size=size)

    def generate_masks(x, y):
        x_mask = tf.zeros([len(x), len(x)])
        y_mask = create_look_ahead_mask(len(y))
        return x, y, x_mask, y_mask

    return (
        tf.data.Dataset.from_tensor_slices((data, data))
        .map(generate_masks)
        .batch(batch_size=batch_size)
    )

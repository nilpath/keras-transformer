import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, masking=False, **kwargs):
        super(SparseCategoricalCrossentropy, self).__init__(**kwargs)
        self._masking = masking

    def call(self, y_true, y_pred):
        if not self._masking:
            return super().call(y_true, y_pred)

        loss = super().call(y_true, y_pred)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

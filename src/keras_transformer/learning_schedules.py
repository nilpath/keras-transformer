import tensorflow as tf


class ModelSizeSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_size, factor=1, warmup_steps=4000):
        super(ModelSizeSchedule, self).__init__()
        self._model_size = model_size
        self._factor = factor
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        factor = tf.cast(self._factor, tf.float32)
        model_size = tf.cast(self._model_size, tf.float32)
        return (
            factor
            * tf.math.rsqrt(model_size)
            * tf.minimum(tf.math.rsqrt(step), step * (self._warmup_steps ** -1.5))
        )

    def get_config(self):
        config = {
            "factor": self._factor,
            "model_size": self._model_size,
            "warmup_steps": self._warmup_steps,
        }
        return config

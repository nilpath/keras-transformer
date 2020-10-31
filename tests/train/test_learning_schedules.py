import tensorflow as tf

from keras_transformer.train.learning_schedules import ModelSizeSchedule


class ModelSizeScheduleTest(tf.test.TestCase):
    def test_works(self):

        steps = tf.range(5, dtype=tf.float32)
        lr_schedule = ModelSizeSchedule(2, factor=1, warmup_steps=2)

        output = lr_schedule(steps)

        self.assertAllClose(output, [0, 0.24999999, 0.49999997, 0.40824828, 0.35355338])

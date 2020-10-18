import tensorflow as tf

from keras_transformer.masking import create_look_ahead_mask


class CreateLookAheadMaskTest(tf.test.TestCase):
    def test_create_mask(self):
        expected_mask = tf.constant(
            [
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        mask = create_look_ahead_mask(3)
        self.assertAllEqual(mask, expected_mask)

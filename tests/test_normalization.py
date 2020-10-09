import tensorflow as tf

from keras_transformer.normalization import LayerNorm


class LayerNormTest(tf.test.TestCase):
    def test_layer_norm(self):
        expected_values = tf.constant(
            [-1.414213, -0.707106, 0.000000, 0.707106, 1.414213], dtype=tf.float32
        )
        norm = LayerNorm(5)
        norm_values = norm(tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32))

        self.assertAllClose(norm_values, expected_values)

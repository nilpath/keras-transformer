import pytest
import tensorflow as tf

from keras_transformer.attention import scaled_dot_product_attention


class ScaledDotProductAttentionTest(tf.test.TestCase):
    def setUp(self):
        super(ScaledDotProductAttentionTest, self).setUp()

    def test_attention_calculation(self):
        query = tf.ones([3, 3])
        key = tf.ones([3, 3])
        value = tf.ones([3, 3])

        output, p_attn = scaled_dot_product_attention(query, key, value)

        self.assertAllEqual(output, tf.ones([3, 3]))
        self.assertAllEqual(p_attn, (tf.ones([3, 3]) * 0.33333334))

    def test_masking_not_implemented(self):
        query = tf.ones([3, 3])
        key = tf.ones([3, 3])
        value = tf.ones([3, 3])

        with pytest.raises(Exception):
            output, p_attn = scaled_dot_product_attention(query, key, value, mask=True)

import pytest
import tensorflow as tf

from keras_transformer.attention import (
    scaled_dot_product_attention, create_look_ahead_mask
)


# skipping the test_session method
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
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

    def test_masking_gets_applied(self):
        query = tf.ones([3, 3])
        key = tf.ones([3, 3])
        value = tf.ones([3, 3])
        mask = create_look_ahead_mask(3)

        output, p_attn = scaled_dot_product_attention(query, key, value, mask=mask)

        self.assertAllEqual(p_attn, tf.constant([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.33333334, 0.33333334, 0.33333334]
        ]))


class CreateLookAheadMaskTest(tf.test.TestCase):
    def test_create_mask(self):
        expected_mask = tf.constant([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        mask = create_look_ahead_mask(3)
        self.assertAllEqual(mask, expected_mask)

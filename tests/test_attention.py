import pytest
import tensorflow as tf
import numpy as np

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
        np.random.seed(42)
        query = np.random.rand(3, 3).astype("float32")
        key = np.random.rand(3, 3).astype("float32")
        value = np.random.rand(3, 3).astype("float32")
        mask = create_look_ahead_mask(3)

        flip_zero_one_func = np.vectorize(lambda x: 0 if x == 1 else 1)
        flipped_mask = flip_zero_one_func(mask)

        output, p_attn = scaled_dot_product_attention(query, key, value, mask=mask)

        expected = tf.reshape(flipped_mask, [-1])
        target = tf.reshape(p_attn, [-1])

        for expected_val, target_val in zip(expected, target):
            self.assertAllGreaterEqual(expected_val, target_val)


class CreateLookAheadMaskTest(tf.test.TestCase):
    def test_create_mask(self):
        expected_mask = tf.constant([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        mask = create_look_ahead_mask(3)
        self.assertAllEqual(mask, expected_mask)

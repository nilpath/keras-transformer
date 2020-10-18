import numpy as np
import tensorflow as tf

from keras_transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from keras_transformer.masking import create_look_ahead_mask


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


class MultiHeadAttentionTest(tf.test.TestCase):
    def test_correct_output_shapes(self):
        d_model = 12
        heads = 3
        batch_size = 2
        seq_len = 3
        embedding_length = 4

        query = tf.ones([batch_size, seq_len, embedding_length])
        key = tf.ones([batch_size, seq_len, embedding_length])
        value = tf.ones([batch_size, seq_len, embedding_length])

        multihead_attention = MultiHeadAttention(d_model, heads)

        output = multihead_attention(query, key, value)

        self.assertShapeEqual(np.zeros((batch_size, seq_len, d_model)), output)

    def test_masking_gets_applied(self):
        d_model = 12
        heads = 3
        batch_size = 2
        seq_len = 10
        embedding_length = 4
        mask = create_look_ahead_mask(seq_len)

        query = tf.ones([batch_size, seq_len, embedding_length])
        key = tf.ones([batch_size, seq_len, embedding_length])
        value = tf.ones([batch_size, seq_len, embedding_length])

        MultiHeadAttention(d_model, heads)(query, key, value, mask)

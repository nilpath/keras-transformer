import numpy as np
import tensorflow as tf

from keras_transformer.embeddings import Embeddings
from keras_transformer.encoding import PositionalEncoding


class PositionalEncodingTest(tf.test.TestCase):
    def test_output_shape(self):
        d_model = 512
        batch_size = 64
        seq_length = 62
        test_input = tf.random.uniform((batch_size, seq_length, d_model))

        encoding = PositionalEncoding(d_model, dropout=0)

        out = encoding(test_input)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

    def test_output_shape_in_sequential(self):
        vocab_size = 8500
        d_model = 512
        batch_size = 64
        seq_length = 62
        test_input = tf.random.uniform(
            (batch_size, seq_length), dtype=tf.int64, minval=0, maxval=200
        )

        blocks = tf.keras.Sequential(
            [
                Embeddings(vocab_size, d_model),
                PositionalEncoding(d_model, dropout=0),
            ]
        )

        out = blocks(test_input)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

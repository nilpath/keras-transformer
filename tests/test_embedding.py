import numpy as np
import tensorflow as tf

from keras_transformer.embeddings import Embeddings


class EmbeddingTest(tf.test.TestCase):
    def test_output_shape(self):
        vocab_size = 8500
        d_model = 512
        batch_size = 64
        seq_length = 62
        test_input = tf.random.uniform(
            (batch_size, seq_length), dtype=tf.int64, minval=0, maxval=200
        )
        embeddings = Embeddings(vocab_size, d_model)

        out = embeddings(test_input)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

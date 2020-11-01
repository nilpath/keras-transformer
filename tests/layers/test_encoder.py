import numpy as np
import tensorflow as tf

from keras_transformer.layers.encoder import Encoder, EncoderLayer


class EncoderTest(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 64
        seq_length = 43
        d_model = 512
        test_input = tf.random.uniform((batch_size, seq_length, d_model))
        encoder = Encoder(EncoderLayer(d_model, 8, 2048, dropout=0), 2)

        out = encoder(test_input, None)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

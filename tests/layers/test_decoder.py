import numpy as np
import tensorflow as tf

from keras_transformer.layers.decoder import Decoder, DecoderLayer


class DecoderTest(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 64
        seq_length = 26
        enc_seq_length = 62
        d_model = 512
        test_input = tf.random.uniform((batch_size, seq_length, d_model))
        enc_output = tf.random.uniform((batch_size, enc_seq_length, d_model))
        decoder = Decoder(DecoderLayer(d_model, 8, 2048, dropout=0), 2)

        out = decoder(test_input, enc_output, None, None)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

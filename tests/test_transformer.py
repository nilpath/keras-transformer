import numpy as np
import tensorflow as tf

from keras_transformer.embeddings import Embeddings
from keras_transformer.encoding import PositionalEncoding
from keras_transformer.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
)


class EncoderDecoderTest(tf.test.TestCase):
    def test_output_shape(self):
        src_vocab = 8500
        tgt_vocab = 8000
        batch_size = 64
        src_seq_length = 62
        tgt_seq_length = 26
        d_model = 512

        src_input = tf.random.uniform(
            (batch_size, src_seq_length), dtype=tf.int64, minval=0, maxval=200
        )

        tgt_input = tf.random.uniform(
            (batch_size, tgt_seq_length), dtype=tf.int64, minval=0, maxval=200
        )

        transformer = EncoderDecoder(
            Encoder(EncoderLayer(d_model, 8, 2048, dropout=0), 2),
            Decoder(DecoderLayer(d_model, 8, 2048, dropout=0), 2),
            tf.keras.Sequential(
                [Embeddings(src_vocab, d_model), PositionalEncoding(d_model, dropout=0)]
            ),
            tf.keras.Sequential(
                [Embeddings(tgt_vocab, d_model), PositionalEncoding(d_model, dropout=0)]
            ),
        )

        out = transformer(src_input, tgt_input, None, None)

        self.assertShapeEqual(np.zeros((batch_size, tgt_seq_length, d_model)), out)


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


class EncoderTest(tf.test.TestCase):
    def test_output_shape(self):
        batch_size = 64
        seq_length = 43
        d_model = 512
        test_input = tf.random.uniform((batch_size, seq_length, d_model))
        encoder = Encoder(EncoderLayer(d_model, 8, 2048, dropout=0), 2)

        out = encoder(test_input, None)

        self.assertShapeEqual(np.zeros((batch_size, seq_length, d_model)), out)

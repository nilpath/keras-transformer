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


def create_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input(shape=(None,), dtype="int64", name="targets")
    inputs_mask = tf.keras.layers.Input(
        shape=(None,), dtype="float32", name="inputs_mask"
    )
    targets_mask = tf.keras.layers.Input(
        shape=(None,), dtype="float32", name="targets_mask"
    )

    transformer = EncoderDecoder(
        Encoder(EncoderLayer(d_model, h, d_ff, dropout), N),
        Decoder(DecoderLayer(d_model, h, d_ff, dropout), N),
        tf.keras.Sequential(
            [Embeddings(src_vocab, d_model), PositionalEncoding(d_model, dropout)]
        ),
        tf.keras.Sequential(
            [Embeddings(tgt_vocab, d_model), PositionalEncoding(d_model, dropout)]
        ),
    )

    x = transformer(inputs, targets, inputs_mask, targets_mask)
    output = tf.keras.layers.Dense(tgt_vocab)(x)

    model = tf.keras.Model([inputs, targets, inputs_mask, targets_mask], output)

    return model

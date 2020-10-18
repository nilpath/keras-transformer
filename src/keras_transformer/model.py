import tensorflow as tf

from keras_transformer.embeddings import Embeddings
from keras_transformer.encoding import PositionalEncoding
from keras_transformer.masking import create_look_ahead_mask, create_padding_mask
from keras_transformer.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
)


class TransformerModel(tf.keras.Model):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        name="transformer",
        **kwargs
    ):
        super(TransformerModel, self).__init__(name=name, **kwargs)
        self.enc_dec = EncoderDecoder(
            Encoder(EncoderLayer(d_model, h, d_ff, dropout), N),
            Decoder(DecoderLayer(d_model, h, d_ff, dropout), N),
            tf.keras.Sequential(
                [Embeddings(src_vocab, d_model), PositionalEncoding(d_model, dropout)]
            ),
            tf.keras.Sequential(
                [Embeddings(tgt_vocab, d_model), PositionalEncoding(d_model, dropout)]
            ),
        )
        self.final_layer = tf.keras.layers.Dense(tgt_vocab)

    def call(self, inputs):
        src, tgt, src_mask, tgt_mask = inputs
        decoder_output = self.enc_dec(src, tgt, src_mask, tgt_mask)
        return self.final_layer(decoder_output)

    def train_step(self, data):
        x, y = data
        src, tgt = x
        src_mask = create_padding_mask(src, 0)
        tgt_mask = create_look_ahead_mask(tf.shape(tgt)[1])

        with tf.GradientTape() as tape:
            y_pred = self([src, tgt, src_mask, tgt_mask], training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

import tensorflow as tf

from keras_transformer.layers.decoder import Decoder, DecoderLayer
from keras_transformer.layers.embeddings import Embeddings
from keras_transformer.layers.encoder import Encoder, EncoderLayer
from keras_transformer.layers.encoding import PositionalEncoding
from keras_transformer.masking import create_look_ahead_mask, create_padding_mask


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

        self.encoder = Encoder(EncoderLayer(d_model, h, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, h, d_ff, dropout), N)

        self.src_embed = tf.keras.Sequential(
            [Embeddings(src_vocab, d_model), PositionalEncoding(d_model, dropout)]
        )

        self.tgt_embed = tf.keras.Sequential(
            [Embeddings(tgt_vocab, d_model), PositionalEncoding(d_model, dropout)]
        )

        self.final_layer = tf.keras.layers.Dense(tgt_vocab)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def call(self, inputs):
        src, tgt, src_mask, tgt_mask = inputs

        memory = self.encode(src, src_mask)
        decoder_output = self.decode(memory, src_mask, tgt, tgt_mask)

        return self.final_layer(decoder_output)

    def train_step(self, data):
        src, tgt = data
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = create_padding_mask(src, 0)
        tgt_pad_mask = create_padding_mask(tgt_in, 0)
        tgt_look_ahead_mask = create_look_ahead_mask(tf.shape(tgt_in)[1])
        tgt_mask = tf.maximum(tgt_pad_mask, tgt_look_ahead_mask)

        with tf.GradientTape() as tape:
            y_pred = self([src, tgt_in, src_mask, tgt_mask], training=True)
            loss = self.compiled_loss(
                tgt_out, y_pred, regularization_losses=self.losses
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(tgt_out, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        src, tgt = data
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = create_padding_mask(src, 0)
        tgt_pad_mask = create_padding_mask(tgt_in, 0)
        tgt_look_ahead_mask = create_look_ahead_mask(tf.shape(tgt_in)[1])
        tgt_mask = tf.maximum(tgt_pad_mask, tgt_look_ahead_mask)

        y_pred = self([src, tgt_in, src_mask, tgt_mask], training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(tgt_out, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(tgt_out, y_pred)
        return {m.name: m.result() for m in self.metrics}

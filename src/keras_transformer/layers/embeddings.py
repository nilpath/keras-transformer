import tensorflow as tf


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab, d_model, name="embeddings", **kwargs):
        super(Embeddings, self).__init__(name=name, **kwargs)
        self.lut = tf.keras.layers.Embedding(vocab, d_model)
        self.d_model = d_model

    def call(self, x):
        return self.lut(x) * tf.math.sqrt(float(self.d_model))

import tensorflow as tf

from keras_transformer.model import create_model


class ModelTest(tf.test.TestCase):
    def test_compile_model(self):
        vocab_size = 11
        model = create_model(vocab_size, vocab_size, N=2)
        model.compile(optimizer='adam')

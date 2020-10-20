import numpy as np
import tensorflow as tf

from keras_transformer.dataset import synthetic_data


class SyntheticDataTest(tf.test.TestCase):
    def test_src_same_as_tgt(self):
        dataset = synthetic_data(20, 30, 10)
        for src, tgt in dataset:
            self.assertAllEqual(src, tgt)

    def test_src_tgt_shapes(self):
        nbatches = 20
        batch_size = 30
        seq_len = 40
        vocab_size = 800
        dataset = synthetic_data(nbatches, batch_size, seq_len, vec_max=vocab_size)
        for src, tgt in dataset:
            self.assertShapeEqual(np.zeros((batch_size, seq_len)), src)
            self.assertShapeEqual(np.zeros((batch_size, seq_len)), tgt)

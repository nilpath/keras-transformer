import tensorflow as tf

from keras_transformer.data.dataset import synthetic_data
from keras_transformer.model import TransformerModel
from keras_transformer.train.losses import SparseCategoricalCrossentropy


class ModelTest(tf.test.TestCase):
    def test_compile_model(self):
        vocab_size = 11
        model = TransformerModel(vocab_size, vocab_size, N=2)
        model.compile(optimizer="adam")

    def test_train_loop(self):
        src_vocab_size = 200
        tgt_vocab_size = 200
        max_length = 15
        train_ds = synthetic_data(20, 30, max_length, vec_min=1, vec_max=src_vocab_size)
        val_ds = synthetic_data(5, 30, max_length, vec_min=1, vec_max=src_vocab_size)

        model = TransformerModel(src_vocab_size, tgt_vocab_size, N=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=SparseCategoricalCrossentropy(masking=True, from_logits=True),
        )

        model.fit(train_ds, epochs=3, validation_data=val_ds)

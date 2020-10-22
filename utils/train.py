from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

from keras_transformer.data.dataset import create_text_dataset
from keras_transformer.learning_schedules import ModelSizeSchedule
from keras_transformer.losses import SparseCategoricalCrossentropy
from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")[:250]  # TODO: use complete corpus
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")[:250]  # TODO: use complete corpus

    src_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/sv"
    )

    tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/en"
    )

    dataset = create_text_dataset(
        source, target, src_encoder, tgt_encoder, batch_size=64, buffer_size=20000
    )

    d_model = 512

    learning_rate = ModelSizeSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    checkpoint_filepath = './output/checkpoints/sv-en-model-{epoch:02d}-{loss:.2f}/checkpoint.ckpt'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    model = TransformerModel(
        src_encoder.vocab_size, tgt_encoder.vocab_size, d_model=d_model
    )
    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(masking=True, from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')]
    )

    model.fit(dataset, epochs=4, callbacks=[checkpoint_callback])

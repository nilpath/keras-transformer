from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

from keras_transformer.data.dataset import create_text_dataset
from keras_transformer.train.learning_schedules import ModelSizeSchedule
from keras_transformer.train.losses import SparseCategoricalCrossentropy
from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")

    train_data = (source[:128], target[:128])
    val_data = (source[5000:5064], target[5000:5064])

    seq_length = 128
    d_model = 512

    src_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/sv"
    )

    tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/en"
    )

    train_dataset = create_text_dataset(
        train_data[0], train_data[1], src_encoder, tgt_encoder, batch_size=64
    )

    val_dataset = create_text_dataset(
        val_data[0], val_data[1], src_encoder, tgt_encoder, batch_size=64
    )

    learning_rate = ModelSizeSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    checkpoint_filepath = './output/checkpoints/sv-en-model/checkpoint.ckpt'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq=1500,
        save_best_only=True)

    model = TransformerModel(
        src_encoder.vocab_size, tgt_encoder.vocab_size, d_model=d_model
    )
    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(masking=True, from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )

    # model_filepath = "./output/checkpoints/saved/sv-en-model-08-1.65/checkpoint.ckpt"
    model_filepath = None
    if model_filepath:
        model.load_weights(model_filepath)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[checkpoint_callback]
    )

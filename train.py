from typing import List

import tensorflow as tf

from keras_transformer.data.dataset import create_text_dataset
from keras_transformer.learning_schedules import ModelSizeSchedule
from keras_transformer.losses import SparseCategoricalCrossentropy
from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath) as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")[:10000]  # TODO: use complete corpus
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")[:10000]  # TODO: use complete corpus

    dataset, src_encoder, tgt_encoder = create_text_dataset(
        source, target, prefix="<SOS>", postfix="<EOS>",
        batch_size=64, buffer_size=20000
    )

    d_model = 512

    learning_rate = ModelSizeSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    model = TransformerModel(
        src_encoder.vocab_size, tgt_encoder.vocab_size, d_model=d_model
    )
    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(masking=True, from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')]
    )

    model.fit(dataset, epochs=10)
    model.save('./output/sv_en_model')

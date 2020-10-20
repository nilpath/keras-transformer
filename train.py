from typing import List

import tensorflow as tf

from keras_transformer.data.dataset import create_text_dataset
from keras_transformer.losses import SparseCategoricalCrossentropy
from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath) as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")[:200]  # TODO: use complete corpus
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")[:200]  # TODO: use complete corpus

    dataset, src_encoder, tgt_encoder = create_text_dataset(
        source, target, prefix="<SOS>", postfix="<EOS>"
    )

    model = TransformerModel(src_encoder.vocab_size, tgt_encoder.vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(masking=True, from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')]
    )

    model.fit(dataset, epochs=1)

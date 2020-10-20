from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds

from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath) as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")[:5000]  # TODO: use complete corpus
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")[:5000]  # TODO: use complete corpus

    tgt_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        target, target_vocab_size=2**13
    )

    src_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        target, target_vocab_size=2**13
    )

    def tokenize(source: tf.Tensor, target: tf.Tensor):
        # TODO: is it better to add <SOS> and <EOS> as text entities first instead?
        src = [src_tokenizer.vocab_size] + src_tokenizer.encode(source.numpy()) + [src_tokenizer.vocab_size+1]
        tgt = [tgt_tokenizer.vocab_size] + tgt_tokenizer.encode(target.numpy()) + [tgt_tokenizer.vocab_size+1]
        return src, tgt

    def tf_tokenize(source: tf.Tensor, target: tf.Tensor):
        src, tgt = tf.py_function(tokenize, [source, target], [tf.int64, tf.int64])
        src.set_shape([None])
        tgt.set_shape([None])
        return src, tgt

    def max_length(source: tf.Tensor, target: tf.Tensor, max_length=40):
        return tf.logical_and(
            tf.size(source) <= max_length,
            tf.size(target) <= max_length,
        )

    ds = (
        tf.data.Dataset.from_tensor_slices((source, target))
        .map(tf_tokenize)
        .filter(max_length)
        .cache()
        .shuffle(20000)
        .padded_batch(64)
    )

    source_vocab_size = src_tokenizer.vocab_size + 2
    target_vocab_size = tgt_tokenizer.vocab_size + 2

    model = TransformerModel(source_vocab_size, target_vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    model.fit(ds, epochs=10)

from random import choice

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_transformer.data.dataset import create_text_dataset, synthetic_data

src_words = "Citronträd och motorolja Världen räcker inte till".split(" ")
tgt_words = "The worlds fastest indian The world is not enough".split(" ")


def generate_corpus(nbatches, batch_size, sentence_length=40):
    src_corpus = []
    tgt_corpus = []
    for _ in range(nbatches * batch_size):
        src_sentence = [choice(src_words) for _ in range(sentence_length)]
        tgt_sentence = [choice(tgt_words) for _ in range(sentence_length)]
        src_corpus.append(" ".join(src_sentence))
        tgt_corpus.append(" ".join(tgt_sentence))

    return src_corpus, tgt_corpus


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


class TextDatasetTest(tf.test.TestCase):
    def test_tokenize_input(self):
        src_corpus = ["Citronträd och motorolja", "Världen räcker inte till"]
        tgt_corpus = ["The worlds fastest indian", "The world is not enough"]

        src_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            src_corpus, target_vocab_size=2 ** 13, reserved_tokens=["<SOS>", "<EOS>"]
        )

        tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            tgt_corpus, target_vocab_size=2 ** 13, reserved_tokens=["<SOS>", "<EOS>"]
        )

        dataset = create_text_dataset(
            src_corpus,
            tgt_corpus,
            src_encoder,
            tgt_encoder,
            batch_size=1,
            shuffle=False,
        )

        for data, src_txt, tgt_txt in zip(dataset, src_corpus, tgt_corpus):
            src, tgt = data
            self.assertAllEqual(src, [src_encoder.encode("<SOS>" + src_txt + "<EOS>")])
            self.assertAllEqual(tgt, [tgt_encoder.encode("<SOS>" + tgt_txt + "<EOS>")])

    def test_bucket_by_sequence_length(self):

        nbatches = 1
        batch_size = 5

        src_512, tgt_512 = generate_corpus(nbatches, batch_size, sentence_length=512)
        src_256, tgt_256 = generate_corpus(nbatches, batch_size, sentence_length=256)
        src_128, tgt_128 = generate_corpus(nbatches, batch_size, sentence_length=128)
        src_64, tgt_64 = generate_corpus(nbatches, batch_size, sentence_length=64)
        src_32, tgt_32 = generate_corpus(nbatches, batch_size, sentence_length=32)
        src_16, tgt_16 = generate_corpus(nbatches, batch_size, sentence_length=16)

        src_corpus = src_512 + src_256 + src_128 + src_64 + src_32 + src_16
        tgt_corpus = tgt_512 + tgt_256 + tgt_128 + tgt_64 + tgt_32 + tgt_16

        src_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            src_corpus, target_vocab_size=2 ** 13, reserved_tokens=["<SOS>", "<EOS>"]
        )

        tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            tgt_corpus, target_vocab_size=2 ** 13, reserved_tokens=["<SOS>", "<EOS>"]
        )

        dataset = create_text_dataset(
            src_corpus,
            tgt_corpus,
            src_encoder,
            tgt_encoder,
            batch_size=batch_size,
            shuffle=False,
        )

        output_shapes = [
            np.zeros((batch_size, 512 + 2)),
            np.zeros((batch_size, 256 + 2)),
            np.zeros((batch_size, 128 + 2)),
            np.zeros((batch_size, 64 + 2)),
            np.zeros((batch_size, 32 + 2)),
            np.zeros((batch_size, 16 + 2)),
        ]

        for data, output_shape in zip(dataset, output_shapes):
            src, tgt = data
            self.assertShapeEqual(output_shape, src)
            self.assertShapeEqual(output_shape, tgt)

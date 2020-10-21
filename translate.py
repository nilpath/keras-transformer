from typing import List

import tensorflow as tf

from keras_transformer.data.dataset import create_text_dataset
from keras_transformer.masking import create_look_ahead_mask, create_padding_mask
from keras_transformer.model import TransformerModel


def load_file(filepath: str) -> List:
    with open(filepath, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def translate(sentence: str, model, src_encoder, tgt_encoder) -> str:

    sentence = "<SOS>" + sentence + "<EOS>"
    src_input = tf.expand_dims(src_encoder.encode(sentence), 0)

    tgt_input = tgt_encoder.encode("<SOS>")
    output = tf.expand_dims(tgt_input, 0)

    for i in range(40):
        src_mask = create_padding_mask(src_input, 0)
        tgt_mask = tf.expand_dims(create_look_ahead_mask(tf.shape(output)[1]), 0)
        pred = model.predict([src_input, output, src_mask, tgt_mask])
        pred = pred[:, -1:, :]  # (batch_size, 1, vocab_size)
        pred_id = tf.cast(tf.argmax(pred, axis=-1), tf.int32)

        if tf.equal(pred_id, tf.expand_dims(tgt_encoder.encode("<EOS>"), 0)):
            return tgt_encoder.decode(tf.squeeze(output, axis=0))

        output = tf.concat([output, pred_id], axis=-1)

    return tgt_encoder.decode(tf.squeeze(output, axis=0))


if __name__ == "__main__":

    target = load_file("./data/europarl-v7/europarl-v7.sv-en.en")[:250]  # TODO: use complete corpus
    source = load_file("./data/europarl-v7/europarl-v7.sv-en.sv")[:250]  # TODO: use complete corpus

    dataset, src_encoder, tgt_encoder = create_text_dataset(
        source, target, prefix="<SOS>", postfix="<EOS>",
        batch_size=64, buffer_size=20000
    )

    d_model = 512

    model_filepath = "./output/checkpoints/sv-en-model-04-7.69/checkpoint.ckpt"
    model = TransformerModel(
        src_encoder.vocab_size, tgt_encoder.vocab_size, d_model=d_model
    )
    model.load_weights(model_filepath)

    translated = translate('jag älskar dig', model, src_encoder, tgt_encoder)
    print(f"translated: {translated}")
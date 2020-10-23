import tensorflow as tf
import tensorflow_datasets as tfds

from keras_transformer.masking import create_look_ahead_mask, create_padding_mask
from keras_transformer.model import TransformerModel


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

    src_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/sv"
    )

    tgt_encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        "./output/subwords/en"
    )

    d_model = 512

    model_filepath = "./output/checkpoints/sv-en-model-08-1.65/checkpoint.ckpt"
    model = TransformerModel(
        src_encoder.vocab_size, tgt_encoder.vocab_size, d_model=d_model
    )
    model.load_weights(model_filepath)

    translated = translate('jag älskar dig', model, src_encoder, tgt_encoder)
    print(f"translated: {translated}")

from typing import List

import tensorflow_datasets as tfds


def tokenize(
    corpus: List[str], reserved_tokens=["<SOS>", "<EOS>"], output_path: str = "tokens"
):

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus, target_vocab_size=2**13, reserved_tokens=reserved_tokens
    )

    tokenizer.save_to_file(output_path)

from typing import List

import tensorflow_datasets as tfds


def tokenize(corpus: List[str], prefix: str, postfix: str, output_path: str = "tokens"):

    reserved_tokens = []
    if prefix is not None:
        corpus = [prefix + c for c in corpus]
        reserved_tokens.append(prefix)

    if postfix is not None:
        corpus = [c + postfix for c in corpus]
        reserved_tokens.append(postfix)

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus, target_vocab_size=2**13, reserved_tokens=reserved_tokens
    )

    tokenizer.save_to_file(output_path)

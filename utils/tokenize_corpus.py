from typing import List
import sys
import getopt

import tensorflow_datasets as tfds


def load_file(filepath: str) -> List:
    with open(filepath, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":

    inputfile = ""
    outputfile = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print("tokenize_corpus.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("test.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    corpus = load_file(inputfile)

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus, target_vocab_size=2 ** 13, reserved_tokens=["<SOS>", "<EOS>"]
    )

    tokenizer.save_to_file(outputfile)

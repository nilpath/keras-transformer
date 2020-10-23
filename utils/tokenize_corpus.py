from typing import List
import sys
import getopt

from keras_transformer.data.tokenizer import tokenize


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
    tokenize(corpus, output_path=outputfile)

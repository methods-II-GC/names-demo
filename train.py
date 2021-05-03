#!/usr/bin/env python
"""Trains Naïve Bayes classifier for pet names."""

import argparse
import csv
import pickle

import model
import util

from typing import List, Tuple


def main(args: argparse.Namespace) -> None:
    x, y = util.read_tsv(args.train)
    classifier = model.NameClassifier()
    classifier.train(x, y)
    with open(args.model, "wb") as sink:
        pickle.dump(classifier, sink)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", help="path to input train TSV")
    parser.add_argument("model", help="path to output model")
    main(parser.parse_args())

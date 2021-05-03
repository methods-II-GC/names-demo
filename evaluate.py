#!/usr/bin/env python
"""Evaluates Naïve Bayes classifier for pet names."""

import argparse
import csv
import pickle

import util


def main(args: argparse.Namespace) -> None:
    x, y = util.read_tsv(args.test)
    with open(args.model, "rb") as source:
        classifier = pickle.load(source)
    correct = 0
    total = 0
    for y, yhat in zip(y, classifier.predict(x)):
        if y == yhat:
            correct += 1
        total += 1
    print(f"Accuracy:\t{correct / total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", help="path to input test TSV")
    parser.add_argument("model", help="path to input model")
    main(parser.parse_args())

#!/usr/bin/env python
"""Randomly splits the NLTK pet names data."""

import argparse
import itertools
import logging
import random

import nltk  # type: ignore

import util


def main(args: argparse.Namespace) -> None:
    # This only needs to be run once, but leaving it here is harmless.
    assert nltk.download("names")
    random.seed(args.seed)
    # This looks strange but this is how the NLTK corpus readers work.
    m_names = nltk.corpus.names.words("male.txt")
    f_names = nltk.corpus.names.words("female.txt")
    # Constructs a data set of tuples where the first element is the name and
    # the second element is the sex label. This is then shuffled, split, and
    # written out in TSV format.
    x = m_names + f_names
    y = itertools.chain(
        itertools.repeat("M", len(m_names)),
        itertools.repeat("F", len(f_names)),
    )
    data = list(zip(x, y))
    random.shuffle(data)
    ten = len(data) // 10
    util.write_tsv(args.test, data[:ten])
    util.write_tsv(args.dev, data[ten:ten + ten])
    util.write_tsv(args.train, data[ten + ten:])


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("train", help="path to output train TSV")
    parser.add_argument("dev", help="path to output dev TSV")
    parser.add_argument("test", help="path to output test TSV")
    main(parser.parse_args())

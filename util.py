"""Helper utilities."""

import csv
import logging

from typing import List, Tuple


def read_tsv(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r") as source:
        tsv_reader = csv.reader(source, delimiter="\t")
        # This is dark magic that turns a list of lists into
        # two separate lists.
        x, y = zip(*tsv_reader)
        return list(x), list(y)


def write_tsv(path: str, data: List[Tuple[str, str]]) -> None:
    with open(path, "w") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        # This just iteratively writes the pairs.
        logging.info("Writing %d examples to %s", len(data), path)
        tsv_writer.writerows(data)

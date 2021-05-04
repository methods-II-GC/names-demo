"""Naïve Bayes classifier for pet names."""


from typing import Any, Dict, List

import sklearn.feature_extraction  # type: ignore
import sklearn.naive_bayes  # type: ignore


class NameClassifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        self.vectorizer = sklearn.feature_extraction.DictVectorizer()
        # The term "Bernoulli" here refers to Bernoulli trials, observations
        # that can either be true or false. Here these correspond to the
        # vectorizer's binarization of the features.
        self.classifier = sklearn.naive_bayes.BernoulliNB()

    def _extract_features(self, name: str) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        features["lastletter"] = name[-1]
        # I convert this to a string because the vectorizer expects string-like
        # features.
        features["length"] = str(len(name))
        # TODO: add more features here.
        return features

    def train(self, x: List[str], y: List[str]):
        xx = self.vectorizer.fit_transform(
            self._extract_features(name) for name in x
        )
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.vectorizer.transform(
            self._extract_features(name) for name in x
        )
        return list(self.classifier.predict(xx))

import pytest
from model import SentimentClassifier


@pytest.fixture(scope="module")
def classifier():
    return SentimentClassifier("./config.yaml")


def test_classifier_positive_sentiment(classifier):
    result = classifier.predict("Hello, my dog is cute and I love this wonderful day!")
    assert result["label"] == "POSITIVE"
    assert 0.5 < result["confidence"] <= 1.0
    assert "POSITIVE" in result["scores"]
    assert "NEGATIVE" in result["scores"]


def test_classifier_negative_sentiment(classifier):
    result = classifier.predict("This service is terrible, slow, and completely broken.")
    assert result["label"] == "NEGATIVE"
    assert 0.5 < result["confidence"] <= 1.0


def test_classifier_handles_long_text_with_truncation(classifier):
    long_text = "I love this product! " * 400  # > 512 tokens
    result = classifier.predict(long_text)
    assert result["label"] in ("POSITIVE", "NEGATIVE")

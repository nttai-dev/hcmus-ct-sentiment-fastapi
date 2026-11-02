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


from unittest.mock import patch
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "endpoints" in data


def test_health_endpoint_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_health_endpoint_unhealthy_when_model_missing():
    with patch.object(main, "classifier", None):
        response = client.get("/health")
        assert response.status_code == 503


def test_predict_endpoint_valid_inputs():
    pos = client.post("/predict", json={"text": "I really enjoyed the class today!"})
    assert pos.status_code == 200
    assert pos.json()["prediction"]["label"] == "POSITIVE"

    neg = client.post("/predict", json={"text": "The movie was boring and a waste of time."})
    assert neg.status_code == 200
    assert neg.json()["prediction"]["label"] == "NEGATIVE"


def test_predict_rejects_empty_or_whitespace_input():
    r1 = client.post("/predict", json={"text": ""})
    assert r1.status_code == 422

    r2 = client.post("/predict", json={"text": "   \n\t  "})
    assert r2.status_code == 422


def test_predict_rejects_missing_field():
    r = client.post("/predict", json={"wrong_field": "hello"})
    assert r.status_code == 422


def test_predict_handles_inference_runtime_error():
    with patch.object(main.classifier, "predict", side_effect=RuntimeError("CUDA out of memory")):
        r = client.post("/predict", json={"text": "Hello world"})
        assert r.status_code == 500
        assert "Inference failed" in r.json()["detail"]


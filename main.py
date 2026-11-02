from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from model import SentimentClassifier

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "FastAPI web service wrapping Hugging Face "
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english for sentiment classification."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    classifier: SentimentClassifier | None = SentimentClassifier("./config.yaml")
except Exception:
    classifier = None


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text to analyze sentiment.",
        examples=["I really enjoyed the class today!"],
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace only.")
        return stripped


class PredictResponse(BaseModel):
    status: str
    model: str
    input_text: str
    prediction: dict[str, Any]


@app.get("/")
def root() -> dict[str, Any]:
    model_name = (
        classifier.model_path
        if classifier is not None
        else "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    return {
        "status": "ok",
        "api_name": "Hugging Face Sentiment Analysis API",
        "model": model_name,
        "description": (
            "Web API providing English sentiment analysis (POSITIVE / NEGATIVE) "
            "powered by Hugging Face DistilBERT."
        ),
        "endpoints": {
            "GET /": "Introduction and API endpoint overview",
            "GET /health": "Operational health check and model readiness status",
            "POST /predict": "Analyze sentiment for an input text payload",
        },
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: sentiment model is not loaded.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": classifier.model_path,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest) -> PredictResponse:
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: sentiment model is not loaded.",
        )
    try:
        result = classifier.predict(request.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc

    return PredictResponse(
        status="success",
        model=classifier.model_path,
        input_text=request.text,
        prediction={
            "label": result["label"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Sentiment Analysis Web API (Practice 01)

This repository contains my submission for **Practice 01: Application Programming Interface** in the Computational Thinking course at VNU-HCM University of Science (FIT-HCMUS). The project wraps a pre-trained Hugging Face sentiment classification model inside a FastAPI web service so client programs can send English sentences over HTTP and get back sentiment predictions as JSON.

## 1. Student information

- **Full name:** Nguyễn Tấn Tài
- **Student ID:** 25127493
- **Class:** 25C09
- **Course:** Computational Thinking
- **Lab instructors:** Nguyen Ngoc Minh Chau, Le Duc Khoan

## 2. Selected Hugging Face model

- **Model name:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Hugging Face page:** [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **Task:** Text classification / Sentiment analysis (`POSITIVE` vs. `NEGATIVE`)

I picked this model because it is fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset and is small enough (~268 MB) to load in a couple of seconds and run on a normal laptop CPU without needing a dedicated GPU.

## 3. System functionality and project layout

The service exposes three HTTP endpoints through FastAPI:

- `GET /` returns basic metadata about the API, the active model name, and a list of available routes.
- `GET /health` checks whether the model weights and tokenizer finished loading into memory. It returns HTTP `200` when ready or `503` if model initialization failed.
- `POST /predict` accepts a JSON object with a `"text"` field, runs inference through `DistilBertForSequenceClassification`, applies `softmax` to the output logits, and returns the predicted label (`POSITIVE` or `NEGATIVE`) along with confidence scores rounded to 4 decimal places.

Input validation is handled by Pydantic before the model is called:
- Missing fields or invalid JSON payloads return HTTP `422 Unprocessable Entity`.
- Empty strings (`""`) or whitespace-only strings (`"   "`) are stripped and rejected with HTTP `422`.
- Long inputs are truncated at `max_length: 512` tokens in the tokenizer so they do not exceed DistilBERT's positional embedding limit.

### Repository structure

```text
.
├── config.yaml          # Model name and max token length
├── main.py              # FastAPI app, request/response schemas, and route handlers
├── model.py             # SentimentClassifier wrapper around Hugging Face tokenizer and model
├── test_api.py          # Client script that tests the running server using `requests`
├── notebook_dev.ipynb   # Notebook version for running on Google Colab or Jupyter
├── requirements.txt     # Python package dependencies
├── README.md            # Project documentation
└── tests/
    └── test_unit.py     # Automated unit and endpoint tests using pytest
```

## 4. Installing dependencies

Make sure you have Python 3.10 or newer installed. Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/hcmus-ct-sentiment-fastapi.git
cd hcmus-ct-sentiment-fastapi

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or on macOS / Linux:
# python3 -m venv .venv
# source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

On the first run, `transformers` will download the model weights (~268 MB) from Hugging Face Hub into your local cache directory (`~/.cache/huggingface`).

## 5. Running the application

Start the FastAPI server with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run `main.py` directly with Python:

```bash
python main.py
```

Once the server is up:
- **API base URL:** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc documentation:** `http://127.0.0.1:8000/redoc`

## 6. Calling the API (example requests and responses)

### 6.1 Run the automated client script (`requests`)

With the server running in one terminal, open a second terminal and run:

```bash
python test_api.py
```

`test_api.py` sends requests to `GET /`, `GET /health`, and `POST /predict` (with positive, negative, product review, and invalid empty inputs) and prints the status codes and JSON responses.

You can also run the unit test suite (which uses FastAPI's `TestClient` and does not require starting Uvicorn separately):

```bash
pytest -v
```

### 6.2 `GET /` (system introduction)

**Request (`curl`):**

```bash
curl -X GET http://127.0.0.1:8000/
```

**Response (`200 OK`):**

```json
{
  "status": "ok",
  "api_name": "Hugging Face Sentiment Analysis API",
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "description": "Web API providing English sentiment analysis (POSITIVE / NEGATIVE) powered by Hugging Face DistilBERT.",
  "endpoints": {
    "GET /": "Introduction and API endpoint overview",
    "GET /health": "Operational health check and model readiness status",
    "POST /predict": "Analyze sentiment for an input text payload"
  }
}
```

### 6.3 `GET /health` (health check)

**Request (`curl`):**

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (`200 OK`):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

### 6.4 `POST /predict` (Input 1: positive sentiment)

**Request (`curl`):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The lecture on FastAPI was clear, practical, and easy to follow!"}'
```

**Request (Python `requests`):**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "The lecture on FastAPI was clear, practical, and easy to follow!"},
)
print(response.status_code)
print(response.json())
```

**Response (`200 OK`):**

```json
{
  "status": "success",
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "input_text": "The lecture on FastAPI was clear, practical, and easy to follow!",
  "prediction": {
    "label": "POSITIVE",
    "confidence": 0.999,
    "scores": {
      "NEGATIVE": 0.001,
      "POSITIVE": 0.999
    }
  }
}
```

### 6.5 `POST /predict` (Input 2: negative sentiment)

**Request (`curl`):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The app kept crashing every five minutes and lost all my saved work."}'
```

**Response (`200 OK`):**

```json
{
  "status": "success",
  "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "input_text": "The app kept crashing every five minutes and lost all my saved work.",
  "prediction": {
    "label": "NEGATIVE",
    "confidence": 0.9998,
    "scores": {
      "NEGATIVE": 0.9998,
      "POSITIVE": 0.0002
    }
  }
}
```

### 6.6 `POST /predict` (Input validation error example)

Sending an empty or whitespace-only string triggers Pydantic validation before model inference runs:

**Request (`curl`):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```

**Response (`422 Unprocessable Entity`):**

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, Input text cannot be empty or whitespace only.",
      "input": "   ",
      "ctx": {
        "error": {}
      }
    }
  ]
}
```

## 7. Demonstration video

> **Note:** Replace `<YOUR_VIDEO_LINK>` below with your public YouTube or Google Drive video link after recording.

[![Watch the Demonstration Video](https://img.shields.io/badge/Watch_Demo_Video-YouTube_%2F_Google_Drive-red?style=for-the-badge&logo=youtube)](<YOUR_VIDEO_LINK>)

- **Direct link:** [Watch Practice 01 Demo Video](<YOUR_VIDEO_LINK>)
- **What the video covers (under 5 minutes):**
  1. Overview of the `distilbert/distilbert-base-uncased-finetuned-sst-2-english` model on Hugging Face.
  2. Walkthrough of `config.yaml`, `model.py`, `main.py`, and `test_api.py`.
  3. Starting the FastAPI server locally with Uvicorn.
  4. Testing `GET /`, `GET /health`, and `POST /predict` with positive, negative, and invalid inputs using `python test_api.py` and Swagger UI (`/docs`).
  5. Explanation of the returned JSON fields (`label`, `confidence`, and `scores`).

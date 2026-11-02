# Sentiment Analysis Web API (Practice 01)

This repository is my submission for Practice 01 (Application Programming Interface) in the Computational Thinking course at VNU-HCM University of Science (FIT-HCMUS).

## 1. Student information

| Field | Information |
| :--- | :--- |
| Full name | Nguyễn Tấn Tài |
| Student ID | 25127493 |
| Class | 25C09 |
| Course | Computational Thinking (Practice 01 - API) |
| Lab instructors | Nguyen Ngoc Minh Chau, Le Duc Khoan |

## 2. The name of the selected model and a link to its Hugging Face page

Model name: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

Hugging Face page: [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

This is a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset for English binary sentiment classification (`POSITIVE` or `NEGATIVE`). I picked it because the weights are relatively small (around 268 MB), so it loads in a couple of seconds and runs fast on a normal laptop CPU without needing a dedicated GPU.

## 3. A brief description of the system's functionality

The system wraps the Hugging Face sentiment model in a FastAPI web application so clients can send English text over HTTP and receive sentiment predictions in JSON format.

I split the code into separate files to keep the model inference logic separate from the API routes:

```text
.
├── config.yaml          # Model path and maximum token length (512)
├── model.py             # SentimentClassifier wrapper around Hugging Face tokenizer and model
├── main.py              # FastAPI app, Pydantic input validation, and route handlers
├── test_api.py          # Client script that tests the API using the `requests` library
├── notebook_dev.ipynb   # Notebook used during development on Google Colab / Jupyter
├── requirements.txt     # Required Python dependencies
├── README.md            # Project documentation
└── tests/
    └── test_unit.py     # Automated unit and endpoint tests for pytest
```

When the app starts, `main.py` reads `config.yaml` and loads `SentimentClassifier` from `model.py` into memory. The API exposes three endpoints:

1. `GET /` returns a short introduction to the API, the model name, and a list of available routes.
2. `GET /health` checks whether the model finished loading. It returns HTTP `200` when the model is ready, or HTTP `503` if model initialization failed.
3. `POST /predict` receives a JSON payload containing a `"text"` field, tokenizes the text, runs inference inside `torch.no_grad()`, applies `softmax` to get probabilities, and returns the predicted label (`POSITIVE` or `NEGATIVE`) along with confidence scores rounded to 4 decimal places.

For input validation and error handling, `PredictRequest` in `main.py` strips leading/trailing spaces and rejects empty or whitespace-only strings (`""` or `"   "`) with HTTP `422 Unprocessable Entity` before calling the model. The tokenizer in `model.py` also truncates inputs longer than 512 tokens so long paragraphs do not crash DistilBERT. If an unexpected runtime error happens during inference, `POST /predict` catches it and returns HTTP `500` with a JSON error message.

## 4. Instructions for installing the required dependencies

You need Python 3.10 or newer installed on your machine. Clone this repository, create a virtual environment, and install the packages listed in `requirements.txt`:

```powershell
git clone https://github.com/nttai-dev/hcmus-ct-sentiment-fastapi.git
cd hcmus-ct-sentiment-fastapi

# Create and activate a virtual environment on Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

If you use macOS or Linux, activate the virtual environment with `source .venv/bin/activate`. On the first run, `transformers` downloads the 268 MB model checkpoint from Hugging Face Hub into your local cache folder.

## 5. Instructions for running the application

Run the FastAPI server locally with Uvicorn:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run `main.py` directly with Python:

```powershell
python main.py
```

Wait until the terminal prints `Application startup complete`. Once it is running, you can access:
- Base API URL at `http://127.0.0.1:8000`
- Interactive Swagger UI at `http://127.0.0.1:8000/docs`
- ReDoc page at `http://127.0.0.1:8000/redoc`

## 6. Instructions for calling the API, including example requests and responses

### 6.1 Testing with the Python `requests` script (`test_api.py`)

Keep the Uvicorn server running in your first terminal, open a second terminal in the project folder, and run:

```powershell
python test_api.py
```

This script uses the `requests` library to call `GET /`, `GET /health`, and `POST /predict` (with positive, negative, product review, and invalid blank inputs) and prints the status codes and JSON responses.

You can also run the automated `pytest` suite at any time with:

```powershell
pytest -v
```

### 6.2 `GET /` (system introduction)

Example request (`curl`):

```bash
curl -X GET http://127.0.0.1:8000/
```

Example response (`200 OK`):

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

### 6.3 `GET /health` (operational status check)

Example request (`curl`):

```bash
curl -X GET http://127.0.0.1:8000/health
```

Example response (`200 OK`):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

### 6.4 `POST /predict` (Input 1: positive sentiment)

Example request (`curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The lecture on FastAPI was clear, practical, and easy to follow!"}'
```

Example request (Python `requests`):

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "The lecture on FastAPI was clear, practical, and easy to follow!"},
)
print(response.status_code)
print(response.json())
```

Example response (`200 OK`):

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

Example request (`curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The app kept crashing every five minutes and lost all my saved work."}'
```

Example response (`200 OK`):

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

### 6.6 `POST /predict` (Validation error on empty/whitespace input)

Example request (`curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```

Example response (`422 Unprocessable Entity`):

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

## 7. A link to the demonstration video

Replace `<YOUR_VIDEO_LINK>` below with the public YouTube or Google Drive URL after recording the demo:

[![Watch the Demonstration Video](https://img.shields.io/badge/Watch_Demo_Video-YouTube_%2F_Google_Drive-red?style=for-the-badge&logo=youtube)](<YOUR_VIDEO_LINK>)

Video link: [Practice 01 Demonstration Video](<YOUR_VIDEO_LINK>)

# Sentiment analysis web API (Practice 01)

I built this project for **Practice 01: Application Programming Interface** in the Computational Thinking course at FIT-HCMUS. It wraps a pre-trained Hugging Face sentiment model in a FastAPI server so you can send an English sentence over HTTP and get back a `POSITIVE` or `NEGATIVE` label with confidence scores in JSON.

## 1. Student information

| Field | Details |
| :--- | :--- |
| Full name | Nguyễn Tấn Tài |
| Student ID | 25127493 |
| Class | 25C09 |
| Course | Computational Thinking |
| Lab instructors | Nguyen Ngoc Minh Chau, Le Duc Khoan |

## 2. Selected model

I used [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face.

It is a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset for binary sentiment classification. I picked it mainly for practical reasons: the weights are around 268 MB, it loads in about two seconds after the first download, and inference runs fast on a regular laptop CPU without needing a GPU.

## 3. How the system works

The codebase is split into a few small files so the model logic stays separate from the HTTP routes:

```text
.
├── config.yaml          # Model checkpoint name and token limit (512)
├── model.py             # SentimentClassifier class (tokenizer + PyTorch inference)
├── main.py              # FastAPI app, Pydantic validation, and route handlers
├── test_api.py          # Script using `requests` to test the live server
├── notebook_dev.ipynb   # Notebook version for Google Colab / Jupyter
├── requirements.txt     # Python dependencies
└── tests/
    └── test_unit.py     # Automated unit tests for pytest
```

When the server starts, `main.py` initializes `SentimentClassifier` using the settings in `config.yaml`. The API has three routes:

1. `GET /` returns a short description of the service, the model name, and the list of endpoints.
2. `GET /health` checks if `SentimentClassifier` loaded properly. It returns HTTP `200` when the model is ready in memory, or `503` if loading failed.
3. `POST /predict` takes a JSON body like `{"text": "..."}`, runs the text through the DistilBERT tokenizer and model inside `torch.no_grad()`, applies `softmax` to convert raw logits into probabilities, and returns both the top label and the individual scores rounded to 4 decimal places.

To keep bad inputs from crashing the model, `PredictRequest` in `main.py` strips whitespace and rejects empty strings (`""` or `"   "`) with HTTP `422` before the tokenizer is ever called. In `model.py`, the tokenizer also sets `truncation=True` and `max_length=512` so long paragraphs do not trigger a tensor size error in DistilBERT.

## 4. Installing dependencies

You will need Python 3.10 or newer. Clone the repository, create a virtual environment, and install the packages from `requirements.txt`:

```powershell
git clone https://github.com/tantai2512/hcmus-ct-sentiment-fastapi.git
cd hcmus-ct-sentiment-fastapi

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

If you are on macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The first time you run the app, `transformers` will download the 268 MB model checkpoint into your local Hugging Face cache.

## 5. Running the application

Start the FastAPI server locally with Uvicorn:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run `main.py` directly:

```powershell
python main.py
```

Once you see `Application startup complete` in the terminal, the server is listening at `http://127.0.0.1:8000`. You can open `http://127.0.0.1:8000/docs` in your browser to test the endpoints through FastAPI's Swagger UI.

## 6. Calling the API

### Using the `requests` test script

While the Uvicorn server is running in one terminal, open a second terminal and run:

```powershell
python test_api.py
```

This script calls `GET /`, `GET /health`, and `POST /predict` (with positive, negative, product review, and invalid blank inputs) and prints the formatted JSON responses. You can also run the automated test suite without starting the server manually by running `pytest -v`.

### Example 1: `GET /` (system overview)

Request:

```bash
curl -X GET http://127.0.0.1:8000/
```

Response (`200 OK`):

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

### Example 2: `GET /health` (health check)

Request:

```bash
curl -X GET http://127.0.0.1:8000/health
```

Response (`200 OK`):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}
```

### Example 3: `POST /predict` with a positive sentence

Request (`curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The lecture on FastAPI was clear, practical, and easy to follow!"}'
```

Equivalent call in Python using `requests`:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "The lecture on FastAPI was clear, practical, and easy to follow!"},
)
print(resp.status_code)
print(resp.json())
```

Response (`200 OK`):

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

### Example 4: `POST /predict` with a negative sentence

Request (`curl`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The app kept crashing every five minutes and lost all my saved work."}'
```

Response (`200 OK`):

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

### Example 5: Validation error on empty input

If you send an empty string or only spaces, Pydantic rejects the request with HTTP `422`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```

Response (`422 Unprocessable Entity`):

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

Replace `<YOUR_VIDEO_LINK>` below with the public YouTube or Google Drive link after recording:

[![Watch the Demo Video](https://img.shields.io/badge/Watch_Demo_Video-YouTube_%2F_Google_Drive-red?style=for-the-badge&logo=youtube)](<YOUR_VIDEO_LINK>)

Direct link: [Practice 01 Demo Video](<YOUR_VIDEO_LINK>)

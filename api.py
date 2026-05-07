# Run with: uvicorn api:app --reload
# API docs available at: http://localhost:8000/docs

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

from src.model import load_model_for_inference, predict

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Fine-tuned BERT model for binary sentiment classification",
    version="1.0.0",
)

# ── Model config ───────────────────────────────────────────────────────────
# Change this to your HuggingFace repo name after training
HF_REPO_NAME = "hsnx000/bert-sentiment-sst2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model once at startup ─────────────────────────────────────────────
# Loading inside the app (not per-request) is critical for performance.
# A cold-loaded model adds 2-3 seconds per request — unacceptable in production.
print(f"Loading model from {HF_REPO_NAME}...")
model, tokenizer = load_model_for_inference(HF_REPO_NAME)
model = model.to(device)
print(f"Model loaded on {device}")


# ── Request / Response schemas ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely wonderful!"}
        }


class BatchPredictRequest(BaseModel):
    texts: list[str]

    class Config:
        json_schema_extra = {
            "example": {"texts": ["Great film!", "Terrible waste of time."]}
        }


class PredictResponse(BaseModel):
    text:       str
    label:      str
    confidence: float
    scores:     dict[str, float]


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check if the API is running and which model is loaded."""
    return {"status": "ok", "model": HF_REPO_NAME, "device": device}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    """
    Predict sentiment for a single text input.
    Returns label (POSITIVE/NEGATIVE), confidence, and full score breakdown.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = predict(request.text, model, tokenizer, device)
    return PredictResponse(text=request.text, **result)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment for a list of texts.
    Processes each one individually — good enough for a portfolio demo.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")

    results = []
    for text in request.texts:
        result = predict(text, model, tokenizer, device)
        results.append(PredictResponse(text=text, **result))
    return results
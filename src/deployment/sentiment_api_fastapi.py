from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing.deployment_tfidf_preprocessing import preprocess_text

# -----------------------------
# Config
# -----------------------------
LABEL_ORDER = ("positive", "negative", "neutral")


# -----------------------------
# Request / Response
# -----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = "svm"


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


# -----------------------------
# Utils
# -----------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _normalize_label(label: Any) -> str | None:
    text = str(label).strip().lower()
    mapping = {
        "-1": "negative",
        "0": "neutral",
        "1": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    return mapping.get(text)


def _to_prediction(label_probs: dict[str, float]):
    total = sum(label_probs.values())
    if total == 0:
        label_probs = {k: 1 / 3 for k in LABEL_ORDER}
    else:
        label_probs = {k: v / total for k, v in label_probs.items()}

    sentiment = max(label_probs, key=label_probs.get)
    confidence = label_probs[sentiment]

    return sentiment, confidence, label_probs


# -----------------------------
# Service
# -----------------------------
class SentimentService:
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text: str):
        cleaned = preprocess_text(text)
        X = self.vectorizer.transform([cleaned])

        # Case 1: has probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_

            label_probs = {}
            for i, cls in enumerate(classes):
                label = _normalize_label(cls)
                if label:
                    label_probs[label] = float(probs[i])

            return _to_prediction(label_probs)

        # Case 2: decision function (SVM)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)

            if scores.ndim == 1:
                scores = np.array([scores])

            probs = _softmax(scores[0])
            classes = self.model.classes_

            label_probs = {}
            for i, cls in enumerate(classes):
                label = _normalize_label(cls)
                if label:
                    label_probs[label] = float(probs[i])

            return _to_prediction(label_probs)

        raise RuntimeError("Unsupported model type")


# -----------------------------
# Load multiple models
# -----------------------------
services = {
    "svm_cleaned": SentimentService(
        Path("model/SVM_tfidf_cleaned_balanced.pkl"),
        Path("model/text_representation_cleaned_balanced_tfidf_vectorizer.pkl"),
    ),
    "svm_with_stopwords": SentimentService(
        Path("model/SVM_tfidf_with_stopwords.pkl"),
        Path(
            "model/text_representation_tfidf_withoutStopwords_RemovePunctuation_vectorizer.pkl"
        ),
    ),
}

app = FastAPI(title="Sentiment API", version="2.0")


@app.on_event("startup")
def load_models():
    for name, service in services.items():
        service.load()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    text = payload.text.strip()
    model_name = payload.model.lower()

    if not text:
        raise HTTPException(400, "Text is empty")

    if model_name not in services:
        raise HTTPException(
            400,
            f"Model '{model_name}' not available. Choose from {list(services.keys())}",
        )

    service = services[model_name]

    sentiment, confidence, probabilities = service.predict(text)

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        probabilities={k: round(v, 4) for k, v in probabilities.items()},
    )

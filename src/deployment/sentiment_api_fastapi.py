from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing.deployment_tfidf_preprocessing import preprocess_text


DEFAULT_MODEL_PATH = Path("model/SVM_tfidf_cleaned.pkl")
DEFAULT_VECTORIZER_PATH = Path("model/text_representation_cleaned_tfidf_vectorizer.pkl")
LABEL_ORDER = ("positive", "negative", "neutral")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["I love this product"])


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    return exps / denom if denom != 0 else np.ones_like(exps) / len(exps)


def _sentiment_from_numeric(score: float) -> str:
    if score > 0.5:
        return "positive"
    if score < -0.5:
        return "negative"
    return "neutral"


def _confidence_from_numeric(score: float, sentiment: str) -> float:
    if sentiment == "neutral":
        confidence = 1.0 - min(abs(score) / 0.5, 1.0)
    else:
        confidence = min(max((abs(score) - 0.5) / 0.5, 0.0), 1.0)
    return float(confidence)


def _normalize_label(label: Any) -> str | None:
    text = str(label).strip().lower()
    numeric_map = {"-1": "negative", "0": "neutral", "1": "positive"}
    alias_map = {
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "positive": "positive",
        "pos": "positive",
    }
    if text in numeric_map:
        return numeric_map[text]
    return alias_map.get(text)


def _normalize_probabilities(label_probs: dict[str, float]) -> dict[str, float]:
    probs = {label: float(label_probs.get(label, 0.0)) for label in LABEL_ORDER}
    total = sum(probs.values())
    if total <= 0:
        return {label: 1.0 / len(LABEL_ORDER) for label in LABEL_ORDER}
    return {label: probs[label] / total for label in LABEL_ORDER}


def _to_prediction(
    label_probs: dict[str, float],
) -> tuple[str, float, dict[str, float]]:
    normalized = _normalize_probabilities(label_probs)
    sentiment = max(normalized, key=normalized.get)
    confidence = float(normalized[sentiment])
    return sentiment, confidence, normalized


def _probabilities_from_numeric_score(score: float) -> dict[str, float]:
    centers = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }
    logits = np.array(
        [-abs(score - centers[label]) for label in ("positive", "negative", "neutral")]
    )
    probs = _softmax(logits)
    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2]),
    }


class SentimentService:
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model: Any | None = None
        self.vectorizer: Any | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {self.vectorizer_path}"
            )

        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)

        with self.vectorizer_path.open("rb") as f:
            self.vectorizer = pickle.load(f)

        if hasattr(self.model, "n_features_in_"):
            vectorized_shape = self.vectorizer.transform(["compatibility check"]).shape[
                1
            ]
            expected_shape = int(self.model.n_features_in_)
            if vectorized_shape != expected_shape:
                raise ValueError(
                    "Model/vectorizer feature mismatch. "
                    f"Model expects {expected_shape} features but vectorizer outputs {vectorized_shape}. "
                    "Regenerate artifacts so both are trained together."
                )

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Artifacts are not loaded")

        cleaned = preprocess_text(text)
        features = self.vectorizer.transform([cleaned])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            classes = self.model.classes_
            label_probs: dict[str, float] = {}
            for idx, cls in enumerate(classes):
                normalized = _normalize_label(cls)
                if normalized is not None:
                    label_probs[normalized] = float(probs[idx])
            return _to_prediction(label_probs)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)
            arr = np.asarray(scores)
            if arr.ndim == 1:
                p = _sigmoid(arr)[0]
                return _to_prediction(
                    {
                        "positive": float(p),
                        "negative": float(1 - p),
                        "neutral": 0.0,
                    }
                )
            probs = _softmax(arr[0])
            classes = self.model.classes_
            label_probs = {}
            for idx, cls in enumerate(classes):
                normalized = _normalize_label(cls)
                if normalized is not None:
                    label_probs[normalized] = float(probs[idx])
            return _to_prediction(label_probs)

        score = float(self.model.predict(features)[0])
        probs = _probabilities_from_numeric_score(score)
        sentiment, confidence, normalized_probs = _to_prediction(probs)
        return sentiment, confidence, normalized_probs


model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
vectorizer_path = Path(os.getenv("VECTORIZER_PATH", str(DEFAULT_VECTORIZER_PATH)))
service = SentimentService(model_path=model_path, vectorizer_path=vectorizer_path)

app = FastAPI(title="Sentiment Prediction API", version="1.0.0")


@app.on_event("startup")
def _startup_load_artifacts() -> None:
    service.load()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must not be empty")

    try:
        sentiment, confidence, probabilities = service.predict(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rounded_probabilities = {
        label: round(float(probabilities.get(label, 0.0)), 4) for label in LABEL_ORDER
    }
    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        probabilities=rounded_probabilities,
    )

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


INPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\sample_2_preprocessed_simple.csv")
OUTPUT_DIR = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs")
PREDICTIONS_PATH = OUTPUT_DIR / "svm_linear_regression_cleaned_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "svm_linear_regression_cleaned_summary.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_NUMBER = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

df = pd.read_csv(INPUT_PATH)
texts = df["preprocessed_review"].fillna("")
labels = df["class_1"].str.strip().str.lower()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X, labels, df.index, test_size=0.25, random_state=42, stratify=labels
)

svm_model = LinearSVC(random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train.toarray(), y_train.map(LABEL_TO_NUMBER))

svm_test_predictions = pd.Series(svm_model.predict(X_test), index=y_test.index)
lr_test_scores = linear_regression_model.predict(X_test.toarray())
lr_test_predictions = pd.Series(
    np.select(
        [lr_test_scores <= -0.33, lr_test_scores >= 0.33],
        ["negative", "positive"],
        default="neutral",
    ),
    index=y_test.index,
)

all_svm_predictions = svm_model.predict(X)
all_lr_scores = linear_regression_model.predict(X.toarray())
all_lr_predictions = np.select(
    [all_lr_scores <= -0.33, all_lr_scores >= 0.33],
    ["negative", "positive"],
    default="neutral",
)

split = pd.Series("train", index=df.index)
split.loc[test_index] = "test"

predictions = pd.DataFrame(
    {
        "row_number": df.index + 1,
        "review_id": df["review_id"],
        "split": split,
        "true_label": labels,
        "svm_label": all_svm_predictions,
        "linear_regression_score": np.round(all_lr_scores, 4),
        "linear_regression_label": all_lr_predictions,
    }
)
predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

summary = {
    "rows_used": int(len(df)),
    "train_rows": int(len(X_train.shape) and X_train.shape[0]),
    "test_rows": int(len(X_test.shape) and X_test.shape[0]),
    "predictions_rows_exported": int(len(predictions)),
    "svm": {
        "accuracy": round(float(accuracy_score(y_test, svm_test_predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, svm_test_predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, svm_test_predictions, labels=LABEL_ORDER).tolist(),
    },
    "linear_regression": {
        "accuracy": round(float(accuracy_score(y_test, lr_test_predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, lr_test_predictions, labels=LABEL_ORDER, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, lr_test_predictions, labels=LABEL_ORDER).tolist(),
    },
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Summary: {SUMMARY_PATH}")

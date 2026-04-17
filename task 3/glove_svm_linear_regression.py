import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


REPRESENTATION_PATH = Path(r"C:\Users\Lenovo\Desktop\tfidf_representation_cleaned_tfidf.xls")
LABELS_PATH = Path(r"C:\Users\Lenovo\Desktop\sample_2.csv")
LABEL_COLUMN = "class_1"
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_NUMBER = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

OUTPUT_DIR = Path(__file__).with_name("outputs")
PREDICTIONS_PATH = OUTPUT_DIR / "tfidf_svm_linear_regression_predictions_simple.csv"
SUMMARY_PATH = OUTPUT_DIR / "tfidf_svm_linear_regression_summary_simple.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

representation = pd.read_csv(REPRESENTATION_PATH)
labels = pd.read_csv(LABELS_PATH)

if len(representation) != len(labels):
    raise ValueError(f"Row mismatch: representation={len(representation)} labels={len(labels)}")

df = representation.copy()
df["review_id"] = labels["review_id"]
df["true_label"] = labels[LABEL_COLUMN].str.strip().str.lower()

X = representation.to_numpy(dtype=np.float32)
y = df["true_label"]

X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42, stratify=y
)

svm_model = LinearSVC(random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train.map(LABEL_TO_NUMBER))

svm_test_predictions = pd.Series(svm_model.predict(X_test), index=y_test.index)
lr_test_scores = linear_regression_model.predict(X_test)
lr_test_predictions = pd.Series(
    np.select(
        [lr_test_scores <= -0.33, lr_test_scores >= 0.33],
        ["negative", "positive"],
        default="neutral",
    ),
    index=y_test.index,
)

all_svm_predictions = svm_model.predict(X)
all_lr_scores = linear_regression_model.predict(X)
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
        "true_label": y,
        "svm_label": all_svm_predictions,
        "linear_regression_score": np.round(all_lr_scores, 4),
        "linear_regression_label": all_lr_predictions,
    }
)
predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

summary = {
    "representation_path": str(REPRESENTATION_PATH),
    "labels_path": str(LABELS_PATH),
    "representation": "tfidf",
    "rows_used": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
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

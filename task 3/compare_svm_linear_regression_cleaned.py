from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


INPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\sample_2_preprocessed_simple.csv")
COMPARISON_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\ml_comparison_cleaned.csv")
PREDICTIONS_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\ml_predictions_cleaned.csv")

df = pd.read_csv(INPUT_PATH)
texts = df["preprocessed_review"].fillna("")
labels = df["class_1"].str.strip().str.lower()

X = TfidfVectorizer().fit_transform(texts)
X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X, labels, df.index, test_size=0.25, random_state=42, stratify=labels
)

svm = LinearSVC(random_state=42, class_weight="balanced")
svm.fit(X_train, y_train)
svm_test = svm.predict(X_test)

lr = LinearRegression()
lr.fit(X_train.toarray(), y_train.map({"negative": -1.0, "neutral": 0.0, "positive": 1.0}))
lr_scores_test = lr.predict(X_test.toarray())
lr_test = np.select(
    [lr_scores_test <= -0.33, lr_scores_test >= 0.33],
    ["negative", "positive"],
    default="neutral",
)

comparison = pd.DataFrame(
    [
        {
            "model": "LinearSVC",
            "representation": "tfidf",
            "preprocessing": "lowercase_remove_punctuation_remove_stopwords",
            "accuracy": round(float(accuracy_score(y_test, svm_test)), 4),
            "macro_f1": round(float(f1_score(y_test, svm_test, labels=["negative", "neutral", "positive"], average="macro", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, svm_test, labels=["negative", "neutral", "positive"]).tolist(),
        },
        {
            "model": "LinearRegression",
            "representation": "tfidf",
            "preprocessing": "lowercase_remove_punctuation_remove_stopwords",
            "accuracy": round(float(accuracy_score(y_test, lr_test)), 4),
            "macro_f1": round(float(f1_score(y_test, lr_test, labels=["negative", "neutral", "positive"], average="macro", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, lr_test, labels=["negative", "neutral", "positive"]).tolist(),
        },
    ]
)
comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8")

all_svm = svm.predict(X)
all_lr_scores = lr.predict(X.toarray())
all_lr = np.select(
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
        "svm_label": all_svm,
        "linear_regression_score": np.round(all_lr_scores, 4),
        "linear_regression_label": all_lr,
    }
)
predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

print(f"Comparison: {COMPARISON_PATH}")
print(f"Predictions: {PREDICTIONS_PATH}")

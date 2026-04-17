import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


INPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\sample_2.csv")
RESULTS_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\preprocessing_model_comparison.csv")
PREDICTIONS_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\preprocessing_model_predictions.csv")
COMBINED_OUTPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\preprocessing_all_outputs.xlsx")

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_NUMBER = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str, remove_punctuation: bool, remove_stopwords: bool) -> str:
    text = str(text).lower()
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    if remove_stopwords:
        words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)


def vader_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


df = pd.read_csv(INPUT_PATH)
df["label"] = df["class_1"].str.strip().str.lower()
df = df[df["label"].isin(LABELS)].reset_index(drop=True)

schemes = {
    "lowercase_only": {"remove_punctuation": False, "remove_stopwords": False},
    "lowercase_remove_punctuation": {"remove_punctuation": True, "remove_stopwords": False},
    "lowercase_remove_punctuation_remove_stopwords": {"remove_punctuation": True, "remove_stopwords": True},
}

train_index, test_index = train_test_split(
    df.index, test_size=0.25, random_state=42, stratify=df["label"]
)

sia = SentimentIntensityAnalyzer()
results = []
predictions_rows = []

for scheme_name, options in schemes.items():
    texts = df["review"].fillna("").map(
        lambda text: clean_text(
            text,
            remove_punctuation=options["remove_punctuation"],
            remove_stopwords=options["remove_stopwords"],
        )
    )

    train_texts = texts.loc[train_index]
    test_texts = texts.loc[test_index]
    y_train = df.loc[train_index, "label"]
    y_test = df.loc[test_index, "label"]

    vader_test = test_texts.map(lambda text: vader_label(sia.polarity_scores(text)["compound"]))
    vader_scores_all = texts.map(lambda text: round(sia.polarity_scores(text)["compound"], 4))
    vader_labels_all = texts.map(lambda text: vader_label(sia.polarity_scores(text)["compound"]))

    results.append(
        {
            "model": "VADER",
            "representation": "lexicon",
            "preprocessing": scheme_name,
            **metrics(y_test, vader_test),
        }
    )

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    x_all = vectorizer.transform(texts)

    svm = LinearSVC(random_state=42, class_weight="balanced")
    svm.fit(x_train, y_train)
    svm_test = svm.predict(x_test)
    svm_all = svm.predict(x_all)

    results.append(
        {
            "model": "LinearSVC",
            "representation": "tfidf",
            "preprocessing": scheme_name,
            **metrics(y_test, svm_test),
        }
    )

    lr = LinearRegression()
    lr.fit(x_train.toarray(), y_train.map(LABEL_TO_NUMBER))
    lr_test_scores = lr.predict(x_test.toarray())
    lr_test = np.select(
        [lr_test_scores <= -0.33, lr_test_scores >= 0.33],
        ["negative", "positive"],
        default="neutral",
    )
    lr_all_scores = lr.predict(x_all.toarray())
    lr_all = np.select(
        [lr_all_scores <= -0.33, lr_all_scores >= 0.33],
        ["negative", "positive"],
        default="neutral",
    )

    results.append(
        {
            "model": "LinearRegression",
            "representation": "tfidf",
            "preprocessing": scheme_name,
            **metrics(y_test, lr_test),
        }
    )

    split = pd.Series("train", index=df.index)
    split.loc[test_index] = "test"
    for i in df.index:
        predictions_rows.append(
            {
                "row_number": i + 1,
                "review_id": df.loc[i, "review_id"],
                "review": df.loc[i, "review"],
                "processed_text": texts.loc[i],
                "preprocessing": scheme_name,
                "split": split.loc[i],
                "true_label": df.loc[i, "label"],
                "vader_score": vader_scores_all.loc[i],
                "vader_label": vader_labels_all.loc[i],
                "svm_label": svm_all[i],
                "linear_regression_score": round(float(lr_all_scores[i]), 4),
                "linear_regression_label": lr_all[i],
            }
        )

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(predictions_rows)

results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

summary = {
    "rows_used": int(len(df)),
    "train_rows": int(len(train_index)),
    "test_rows": int(len(test_index)),
    "preprocessing_schemes": list(schemes.keys()),
    "models": ["VADER", "LinearSVC", "LinearRegression"],
}
summary_df = pd.DataFrame(
    [
        {
            "rows_used": int(len(df)),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "preprocessing_schemes": ", ".join(schemes.keys()),
            "models": "VADER, LinearSVC, LinearRegression",
        }
    ]
)

with pd.ExcelWriter(COMBINED_OUTPUT_PATH) as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    results_df.to_excel(writer, sheet_name="comparison", index=False)
    predictions_df.to_excel(writer, sheet_name="predictions", index=False)

print(json.dumps(summary, indent=2))
print(f"Results: {RESULTS_PATH}")
print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Combined output: {COMBINED_OUTPUT_PATH}")

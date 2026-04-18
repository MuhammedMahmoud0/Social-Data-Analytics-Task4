import re
from pathlib import Path

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


FILE_PATH = Path(r"C:\Users\Lenovo\Desktop\sample.csv")
OUTPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\sample_compared_simple.csv")
LEXICON = {
    "amazing": 4,
    "awesome": 4,
    "awful": -4,
    "bad": -2,
    "boring": -3,
    "disappointing": -3,
    "enjoyable": 3,
    "excellent": 5,
    "fantastic": 4,
    "good": 2,
    "great": 3,
    "horrible": -4,
    "interesting": 2,
    "love": 3,
    "masterpiece": 5,
    "mediocre": -2,
    "poor": -2,
    "slow": -2,
    "terrible": -4,
    "wonderful": 4,
    "worst": -5,
}
NEGATIONS = {"no", "not", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't"}


def vader_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def afinn_result(text: str) -> tuple[float, str]:
    score = 0
    negate = False
    for word in re.findall(r"[a-z']+", str(text).lower()):
        if word in NEGATIONS:
            negate = True
            continue
        if word in LEXICON:
            value = -LEXICON[word] if negate else LEXICON[word]
            score += value
            negate = False
    if score > 0.5:
        return score, "positive"
    if score < -0.5:
        return score, "negative"
    return score, "neutral" 


df = pd.read_csv(FILE_PATH)
sia = SentimentIntensityAnalyzer()

vader_compound = []
vader_labels = []
afinn_scores = []
afinn_labels = []

for text in df["review"].fillna(""):
    vader = sia.polarity_scores(str(text))
    afinn_score, afinn_label = afinn_result(text)

    vader_compound.append(round(vader["compound"], 4))
    vader_labels.append(vader_label(vader["compound"]))
    afinn_scores.append(round(afinn_score, 4))
    afinn_labels.append(afinn_label)

df["vader_compound"] = vader_compound
df["vader_label"] = vader_labels
df["afinn_score"] = afinn_scores
df["afinn_label"] = afinn_labels
df["models_agree"] = df["vader_label"] == df["afinn_label"]

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Output: {OUTPUT_PATH}")

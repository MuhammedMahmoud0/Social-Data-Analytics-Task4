import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords


INPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\sample_2.csv")
OUTPUT_PATH = Path(r"C:\Users\Lenovo\Desktop\task 3\outputs\sample_2_preprocessed_simple.csv")
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


df = pd.read_csv(INPUT_PATH)
df["preprocessed_review"] = df["review"].fillna("").map(clean_text)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Output: {OUTPUT_PATH}")

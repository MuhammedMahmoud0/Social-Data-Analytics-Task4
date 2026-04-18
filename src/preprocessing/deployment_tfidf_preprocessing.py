from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.text_preprocessing import extract_tags, preprocess


def full_preprocess_args() -> SimpleNamespace:
    """Enable all preprocessing steps from text_preprocessing.py."""
    return SimpleNamespace(
        remove_html=True,
        remove_urls=True,
        remove_punctuation=True,
        lowercase=True,
        remove_stopwords=True,
        remove_numbers=True,
        remove_mentions=True,
        remove_hashtags=True,
        remove_single_chars=True,
        lemmatize=True,
        fix_spelling=True,
        extract_tags=True,
    )


def preprocess_text(text: str, preprocess_args: SimpleNamespace | None = None) -> str:
    args = preprocess_args or full_preprocess_args()
    return preprocess(text, args)


def preprocess_text_with_tags(
    text: str, preprocess_args: SimpleNamespace | None = None
) -> tuple[str, list[str]]:
    args = preprocess_args or full_preprocess_args()
    raw_text = "" if text is None else str(text)
    _, tags = extract_tags(raw_text)
    cleaned = preprocess(raw_text, args)
    return cleaned, tags


def preprocess_texts(
    texts: Iterable[str], include_tags: bool = True
) -> tuple[list[str], list[list[str]] | None]:
    args = full_preprocess_args()
    cleaned_texts: list[str] = []
    tags_per_text: list[list[str]] = []

    for text in texts:
        if include_tags:
            cleaned, tags = preprocess_text_with_tags(text, args)
            cleaned_texts.append(cleaned)
            tags_per_text.append(tags)
        else:
            cleaned_texts.append(preprocess_text(text, args))

    return cleaned_texts, tags_per_text if include_tags else None


def fit_tfidf(
    cleaned_texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    return vectorizer, tfidf_matrix


def transform_with_saved_vectorizer(
    texts: list[str], vectorizer_path: str | Path
) -> tuple[csr_matrix, list[str]]:
    vectorizer_path = Path(vectorizer_path)
    with vectorizer_path.open("rb") as f:
        vectorizer: TfidfVectorizer = pickle.load(f)

    cleaned_texts, _ = preprocess_texts(texts, include_tags=False)
    matrix = vectorizer.transform(cleaned_texts)
    return matrix, cleaned_texts


def build_tfidf_artifacts(
    input_csv: str | Path,
    text_column: str,
    preprocessed_output: str | Path,
    matrix_output: str | Path,
    vectorizer_output: str | Path,
    include_tags: bool,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
) -> None:
    input_csv = Path(input_csv)
    preprocessed_output = Path(preprocessed_output)
    matrix_output = Path(matrix_output)
    vectorizer_output = Path(vectorizer_output)

    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"CSV must contain a column named '{text_column}'")

    cleaned_texts, tags_per_text = preprocess_texts(df[text_column].fillna(""), include_tags)
    vectorizer, tfidf_matrix = fit_tfidf(
        cleaned_texts=cleaned_texts,
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
    )

    for path in [preprocessed_output, matrix_output, vectorizer_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_df = df.copy()
    output_df["cleaned_review"] = cleaned_texts
    if include_tags and tags_per_text is not None:
        output_df["tags"] = [", ".join(tags) for tags in tags_per_text]

    output_df.to_csv(preprocessed_output, index=False, encoding="utf-8")

    save_npz(matrix_output, tfidf_matrix)
    with vectorizer_output.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved cleaned data: {preprocessed_output}")
    print(f"Saved TF-IDF matrix: {matrix_output}")
    print(f"Saved TF-IDF vectorizer: {vectorizer_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full preprocessing pipeline and fit TF-IDF artifacts for deployment."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--text_column", default="review", help="Text column name")
    parser.add_argument(
        "--preprocessed_output",
        default="output/preprocessed_tfidf_all_steps.csv",
        help="Output CSV with cleaned text",
    )
    parser.add_argument(
        "--vectorizer_output",
        default="output/tfidf_all_steps_vectorizer.pkl",
        help="Output path for pickled TF-IDF vectorizer",
    )
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument(
        "--skip_tags",
        action="store_true",
        help="Skip tag extraction output column",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_tfidf_artifacts(
        input_csv=args.input,
        text_column=args.text_column,
        preprocessed_output=args.preprocessed_output,
        matrix_output=args.matrix_output,
        vectorizer_output=args.vectorizer_output,
        include_tags=not args.skip_tags,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
    )


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from textblob import Word
from symspellpy import SymSpell, Verbosity

stop = set(stopwords.words("english"))

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = "en-80k.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def remove_html(text):
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def lowercase(text):
    return text.lower()


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop])


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


def remove_single_chars(text):
    return re.sub(r"\b\w\b", "", text)


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_text(text):
    return " ".join([Word(word).lemmatize() for word in text.split()])


def fix_spelling(text):
    corrected_words = []

    for word in text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def extract_tags(text):
    tags = re.findall(r"#(\w+)", text)
    return text, tags


def preprocess(text, args):

    if not isinstance(text, str):
        return ""

    if args.remove_html:
        text = remove_html(text)

    if args.remove_urls:
        text = remove_urls(text)

    if args.lowercase:
        text = lowercase(text)

    if args.remove_punctuation:
        text = remove_punctuation(text)

    if args.remove_numbers:
        text = remove_numbers(text)

    if args.remove_mentions:
        text = remove_mentions(text)

    if args.remove_hashtags:
        text = remove_hashtags(text)

    if args.remove_stopwords:
        text = remove_stopwords(text)

    if args.remove_single_chars:
        text = remove_single_chars(text)

    if args.fix_spelling:
        text = fix_spelling(text)

    if args.lemmatize:
        text = lemmatize_text(text)

    text = remove_extra_spaces(text)

    return text


def parse_args():

    parser = argparse.ArgumentParser(
        description="Configurable Text Preprocessing Pipeline"
    )

    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    parser.add_argument("--remove_html", action="store_true")
    parser.add_argument("--remove_urls", action="store_true")
    parser.add_argument("--remove_punctuation", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--remove_numbers", action="store_true")
    parser.add_argument("--remove_mentions", action="store_true")
    parser.add_argument("--remove_hashtags", action="store_true")
    parser.add_argument("--remove_single_chars", action="store_true")

    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--fix_spelling", action="store_true")
    parser.add_argument("--extract_tags", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    df = pd.read_csv(args.input)

    if "review" not in df.columns:
        raise ValueError("CSV must contain a column named 'review'")

    df["cleaned_review"] = df["review"].apply(lambda x: preprocess(x, args))
    # df["tags"] = df["review"].apply(
    #     lambda x: extract_tags(x)[1] if args.extract_tags else []
    # )

    df = df[~df["cleaned_review"].isna() & (df["cleaned_review"].str.strip() != "")]

    df.to_csv(args.output, index=False)

    print("Preprocessing completed.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

MAX_WORKERS = 10
MAX_REVIEW_PAGES = 2          # increase to get more data
TARGET_REVIEWS = 5000         # total desired
RETRY_COUNT = 3

TARGET_PER_CLASS = TARGET_REVIEWS // 3


# -----------------------------
# Rating → Category
# -----------------------------
def rating_category(rating):
    if pd.isna(rating):
        return None
    elif rating >= 7:
        return "positive"
    elif rating <= 4:
        return "negative"
    else:
        return "neutral"


# -----------------------------
# Safe request with retry
# -----------------------------
def safe_request(url, params):
    for _ in range(RETRY_COUNT):
        try:
            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                return r.json()

            elif r.status_code == 429:
                time.sleep(1)  # rate limit

            else:
                time.sleep(0.5)

        except requests.exceptions.RequestException:
            time.sleep(0.5)

    return None


# -----------------------------
# Step 1: Get genres
# -----------------------------
genre_data = safe_request(
    f"{BASE_URL}/genre/movie/list",
    {"api_key": API_KEY, "language": "en-US"},
)

genre_map = {g["id"]: g["name"] for g in genre_data["genres"]}
print("✅ Genres loaded")


# -----------------------------
# Step 2: Get movies
# -----------------------------
movies = []

for page in range(1, 300):  # increase pages for better balance
    data = safe_request(
        f"{BASE_URL}/movie/popular",
        {"api_key": API_KEY, "language": "en-US", "page": page},
    )

    if data and "results" in data:
        movies.extend(data["results"])

print(f"🎬 Movies collected: {len(movies)}")


# -----------------------------
# Step 3: Fetch reviews
# -----------------------------
def fetch_reviews(movie):
    results = []

    movie_id = movie["id"]
    movie_title = movie["title"]

    genres = [genre_map[g] for g in movie["genre_ids"] if g in genre_map]
    genre_string = ", ".join(genres)

    for page in range(1, MAX_REVIEW_PAGES + 1):

        data = safe_request(
            f"{BASE_URL}/movie/{movie_id}/reviews",
            {"api_key": API_KEY, "page": page},
        )

        if not data or "results" not in data or len(data["results"]) == 0:
            break

        for review in data["results"]:
            results.append(
                {
                    "movie_id": movie_id,
                    "movie_title": movie_title,
                    "category": genre_string,
                    "review_id": review["id"],
                    "author": review["author"],
                    "review": review["content"],
                    "rating": review["author_details"]["rating"],
                    "created_at": review["created_at"],
                }
            )

    return results


# -----------------------------
# Step 4: Balanced collection
# -----------------------------
class_buckets = {
    "positive": [],
    "neutral": [],
    "negative": [],
}


def is_done():
    return all(len(class_buckets[c]) >= TARGET_PER_CLASS for c in class_buckets)


print("🚀 Collecting balanced reviews...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_reviews, movie) for movie in movies]

    for future in tqdm(as_completed(futures), total=len(futures)):
        results = future.result()

        for r in results:
            label = rating_category(r["rating"])

            if label is None:
                continue

            if len(class_buckets[label]) < TARGET_PER_CLASS:
                class_buckets[label].append(r)

        if is_done():
            break


# -----------------------------
# Step 5: Combine + Save
# -----------------------------
balanced_data = (
    class_buckets["positive"]
    + class_buckets["neutral"]
    + class_buckets["negative"]
)

df = pd.DataFrame(balanced_data)
# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n📊 Final class distribution:")
for k, v in class_buckets.items():
    print(f"{k}: {len(v)}")

print("Total:", len(df))

df.to_csv("tmdb_reviews_balanced.csv", index=False)

print("✅ Saved to tmdb_reviews_balanced.csv")

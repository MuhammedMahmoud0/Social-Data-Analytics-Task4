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
MAX_REVIEW_PAGES = 1
TARGET_REVIEWS = 10000
RETRY_COUNT = 3


# -----------------------------
# Helper: Safe request with retry
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
    f"{BASE_URL}/genre/movie/list", {"api_key": API_KEY, "language": "en-US"}
)

genre_map = {g["id"]: g["name"] for g in genre_data["genres"]}
print("Genres loaded")


# -----------------------------
# Step 2: Get movies
# -----------------------------
movies = []

for page in range(1, 101):  # reduce pages for speed
    data = safe_request(
        f"{BASE_URL}/movie/popular",
        {"api_key": API_KEY, "language": "en-US", "page": page},
    )
    if data and "results" in data:
        movies.extend(data["results"])

print(f"Movies collected: {len(movies)}")


# -----------------------------
# Step 3: Worker function
# -----------------------------
def fetch_reviews(movie):
    results = []

    movie_id = movie["id"]
    movie_title = movie["title"]

    genres = [genre_map[g] for g in movie["genre_ids"] if g in genre_map]
    genre_string = ", ".join(genres)

    for page in range(1, MAX_REVIEW_PAGES + 1):

        data = safe_request(
            f"{BASE_URL}/movie/{movie_id}/reviews", {"api_key": API_KEY, "page": page}
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
# Step 4: Multithreading
# -----------------------------
reviews_data = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_reviews, movie) for movie in movies]

    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        reviews_data.extend(result)

        # Early stop
        if len(reviews_data) >= TARGET_REVIEWS:
            break


# Trim to exact size
reviews_data = reviews_data[:TARGET_REVIEWS]


# -----------------------------
# Step 5: Save
# -----------------------------
df = pd.DataFrame(reviews_data)

print("Total reviews:", len(df))

df.to_csv("tmdb_reviews_fast.csv", index=False)

print("Done.")

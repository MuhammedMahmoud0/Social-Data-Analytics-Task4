# Social Data Analytics - Sentiment Deployment

This project contains preprocessing, TF-IDF feature preparation, a FastAPI prediction service, and a Streamlit web app frontend.

## Project Structure

- `src/preprocessing/`
  - `text_preprocessing.py`: full text cleaning pipeline
  - `deployment_tfidf_preprocessing.py`: runs all preprocessing steps and builds TF-IDF artifacts
- `src/deployment/`
  - `sentiment_api_fastapi.py`: FastAPI backend (`POST /predict`)
  - `streamlit_sentiment_app.py`: Streamlit frontend
- `data/lexicons/`: AFINN and SymSpell dictionaries
- `data/raw/`: sample/raw data files
- `output/`: model artifacts and prediction outputs

## 1. Setup

### Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Download required NLTK data

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## 2. Build TF-IDF Artifacts

Run preprocessing + TF-IDF generation (all cleaning steps enabled):

```powershell
python -m src.preprocessing.deployment_tfidf_preprocessing --input data/raw/sample_2.csv --text_column review
```

This creates:
- `output/preprocessed_tfidf_all_steps.csv`
- `output/tfidf_all_steps_vectorizer.pkl`

## 3. Ensure Model Artifact Exists

The API expects:
- `output/linear_regression_tfidf_cleaned.pkl`
- `output/tfidf_all_steps_vectorizer.pkl`

If the model pickle is missing, create it from your notebook/training step first.

## 4. Run FastAPI

```powershell
uvicorn src.deployment.sentiment_api_fastapi:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8080/health
```

Prediction request example:

```powershell
curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" -d "{\"text\":\"I love this product\"}"
```

Expected response shape:

```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "probabilities": {
    "positive": 0.94,
    "negative": 0.03,
    "neutral": 0.03
  }
}
```

## 5. Run Streamlit Web App

In a new terminal (while API is running):

```powershell
streamlit run src/deployment/streamlit_sentiment_app.py
```

Then open the shown local Streamlit URL in your browser.

## 6. Optional Environment Variables

You can override artifact/API paths:

- `MODEL_PATH` for the model pickle path
- `VECTORIZER_PATH` for the vectorizer pickle path
- `PREDICT_API_URL` for Streamlit API endpoint

Example:

```powershell
$env:MODEL_PATH="output/linear_regression_tfidf_cleaned.pkl"
$env:VECTORIZER_PATH="output/tfidf_all_steps_vectorizer.pkl"
$env:PREDICT_API_URL="http://127.0.0.1:8000/predict"
```

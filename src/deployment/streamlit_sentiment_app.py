from __future__ import annotations

import os
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DEFAULT_API_URL = os.getenv("PREDICT_API_URL", "http://127.0.0.1:8080/predict")

st.set_page_config(page_title="Sentiment Predictor", layout="centered")

st.title("Sentiment Prediction App")

# -----------------------------
# Inputs
# -----------------------------
api_url = st.text_input("API URL", value=DEFAULT_API_URL)

text_input = st.text_area(
    "Enter review text",
    height=150,
    placeholder="I love this product",
)

model_choice = st.selectbox(
    "Select model",
    ["SVM_Cleaned", "SVM_with_Stopwords"],
)

compare_mode = st.checkbox("Compare both models")

predict_clicked = st.button("Predict", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if predict_clicked:
    text = text_input.strip()

    if not text:
        st.error("Please enter text.")
    else:
        models = (
            ["svm_cleaned", "svm_with_stopwords"] if compare_mode else [model_choice]
        )

        for model in models:
            with st.spinner(f"Running {model}..."):

                try:
                    response = requests.post(
                        api_url,
                        json={"text": text, "model": model},
                        timeout=30,
                    )
                except requests.RequestException as e:
                    st.error(f"API error: {e}")
                    continue

                if response.status_code != 200:
                    st.error(response.text)
                    continue

                data = response.json()

                sentiment = data["sentiment"]
                confidence = data["confidence"]
                probs = data["probabilities"]

                st.subheader(f"Model: {model.upper()}")

                col1, col2 = st.columns(2)
                col1.metric("Sentiment", sentiment)
                col2.metric("Confidence", f"{confidence:.2%}")

                st.write("Probabilities:")
                st.write(probs)

                st.progress(
                    probs["positive"], text=f"Positive: {probs['positive']:.2%}"
                )
                st.progress(
                    probs["negative"], text=f"Negative: {probs['negative']:.2%}"
                )
                st.progress(probs["neutral"], text=f"Neutral: {probs['neutral']:.2%}")

                st.divider()

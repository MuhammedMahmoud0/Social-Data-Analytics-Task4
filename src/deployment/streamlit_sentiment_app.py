from __future__ import annotations

import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("PREDICT_API_URL", "http://127.0.0.1:8080/predict")


st.set_page_config(page_title="Sentiment Predictor", page_icon="S", layout="centered")

st.title("Sentiment Prediction")
st.caption("Frontend for the FastAPI sentiment endpoint")

api_url = st.text_input("Predict endpoint URL", value=DEFAULT_API_URL)
text_input = st.text_area("Enter review text", height=180, placeholder="I love this product")

predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=True)

if predict_clicked:
    text = text_input.strip()
    if not text:
        st.error("Please enter text before predicting.")
    else:
        with st.spinner("Calling API..."):
            try:
                response = requests.post(api_url, json={"text": text}, timeout=30)
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")
            else:
                if response.status_code != 200:
                    try:
                        details = response.json()
                    except ValueError:
                        details = response.text
                    st.error(f"API error ({response.status_code}): {details}")
                else:
                    data = response.json()
                    sentiment = str(data.get("sentiment", "unknown"))
                    confidence = float(data.get("confidence", 0.0))
                    probabilities = data.get("probabilities", {})

                    pos_prob = float(probabilities.get("positive", 0.0))
                    neg_prob = float(probabilities.get("negative", 0.0))
                    neu_prob = float(probabilities.get("neutral", 0.0))

                    col1, col2 = st.columns(2)
                    col1.metric("Sentiment", sentiment)
                    col2.metric("Confidence", f"{confidence:.2%}")

                    st.subheader("Per-label probabilities")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Positive", f"{pos_prob:.2%}")
                    p2.metric("Negative", f"{neg_prob:.2%}")
                    p3.metric("Neutral", f"{neu_prob:.2%}")

                    st.progress(min(max(pos_prob, 0.0), 1.0), text=f"Positive: {pos_prob:.2%}")
                    st.progress(min(max(neg_prob, 0.0), 1.0), text=f"Negative: {neg_prob:.2%}")
                    st.progress(min(max(neu_prob, 0.0), 1.0), text=f"Neutral: {neu_prob:.2%}")

                    if sentiment == "positive":
                        st.success("The model predicts a positive sentiment.")
                    elif sentiment == "negative":
                        st.error("The model predicts a negative sentiment.")
                    else:
                        st.info("The model predicts a neutral sentiment.")

st.divider()
st.markdown("Run API: uvicorn src.deployment.sentiment_api_fastapi:app --reload")
st.markdown("Run app: streamlit run src/deployment/streamlit_sentiment_app.py")

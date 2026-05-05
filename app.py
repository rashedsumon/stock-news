import streamlit as st
from data_loader import load_data
from model import train_model, load_model, predict
from utils import clean_text

st.set_page_config(page_title="Stock News Predictor", layout="centered")

st.title("📈 Stock Market Prediction from News")

# Load or train model
model, vectorizer = load_model()

if model is None:
    st.info("Training model... (first run only)")
    df = load_data()
    model, vectorizer = train_model(df)
    st.success("Model trained successfully!")

# User input
st.subheader("Enter News Headlines")

user_input = st.text_area(
    "Paste multiple news headlines (similar to Top1–Top25):"
)

if st.button("Predict Market Direction"):
    if user_input.strip() == "":
        st.warning("Please enter some news.")
    else:
        cleaned = clean_text(user_input)
        pred = predict(cleaned, model, vectorizer)

        if pred == 1:
            st.success("📈 Market Prediction: UP")
        else:
            st.error("📉 Market Prediction: DOWN")
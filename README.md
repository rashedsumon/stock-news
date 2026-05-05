# 📈 Stock News Prediction App

This app predicts stock market direction based on daily news headlines.

## 🚀 Features
- Uses 25 news headlines per day
- TF-IDF + Logistic Regression model
- Streamlit web interface
- Auto dataset download using kagglehub

## 🧠 Input
- News headlines (Top1–Top25)

## 🎯 Output
- 1 → Market Up
- 0 → Market Down

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
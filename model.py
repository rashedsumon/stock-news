import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/model.pkl"

def train_model(df):
    X = df["combined_news"]
    y = df["Label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # ✅ Safe save
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, vectorizer), f)

    return model, vectorizer


def load_model():
    # ❗ If file doesn't exist → return None
    if not os.path.exists(MODEL_PATH):
        return None, None

    # ❗ If file is empty → ignore it
    if os.path.getsize(MODEL_PATH) == 0:
        return None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            model, vectorizer = pickle.load(f)
        return model, vectorizer

    except (EOFError, pickle.UnpicklingError):
        # ❗ corrupted file → ignore and retrain
        return None, None


def predict(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction
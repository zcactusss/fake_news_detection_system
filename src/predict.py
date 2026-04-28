import pickle
from src.preprocess import clean_text

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    confidence = max(prob) * 100

    if prediction == 1:
        return f"Real News ({confidence:.2f}%)"
    else:
        return f"Fake News ({confidence:.2f}%)"

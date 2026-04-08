import csv
import re
from pathlib import Path

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).with_name("sms_spam.csv")
USER_NAME = "Akash"
PROJECT_NAME = "Akash Task - Spam Detector"

app = Flask(__name__)


def load_data(data_path):
    messages = []
    labels = []
    with open(data_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            messages.append(row["message"].strip())
            labels.append(row["label"].strip())
    return messages, labels


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline():
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42, max_iter=500)),
        ]
    )


def train_model(messages, labels):
    cleaned = [clean_text(message) for message in messages]
    model = build_pipeline()
    model.fit(cleaned, labels)
    return model


def predict_message(model, message):
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    score = max(model.predict_proba([cleaned])[0])
    return label, score


messages, labels = load_data(DATA_FILE)
model = train_model(messages, labels)


@app.route("/", methods=["GET", "POST"])
def index():
    message_text = ""
    prediction = None
    confidence = None

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()
        if message_text:
            prediction, confidence = predict_message(model, message_text)
            confidence = f"{confidence:.2f}"

    return render_template(
        "index.html",
        project_name=PROJECT_NAME,
        user_name=USER_NAME,
        message=message_text,
        prediction=prediction,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)

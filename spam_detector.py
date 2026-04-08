import csv
import re
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).with_name("sms_spam.csv")


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
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, digits=3))
    return model


def predict_message(model, message):
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    score = max(model.predict_proba([cleaned])[0])
    return label, score


def sample_prompt():
    return (
        "Enter a message and press Enter. Type 'quit' or 'exit' to stop.\n"
        "Example: Free entry in 2 a weekly competition to win FA Cup final tickets."
    )


def run():
    messages, labels = load_data(DATA_FILE)
    model = train_model(messages, labels)

    print("\nThis model uses TF-IDF text vectorization and logistic regression.")
    print("It learns from a simple spam/ham sample dataset and predicts new text.")
    print(sample_prompt())

    while True:
        user_text = input("\nYour message: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Goodbye! Keep an eye out for tricky spam.")
            break
        if not user_text:
            print("Please type a message before checking.")
            continue
        label, confidence = predict_message(model, user_text)
        print(f"Prediction: {label.upper()} (confidence {confidence:.2f})")


if __name__ == "__main__":
    run()

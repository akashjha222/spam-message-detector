# Spam Message Detection

This project shows a simple spam detector built in Python.
It uses a small dataset of spam and ham messages, converts text into TF-IDF features, and trains a logistic regression classifier.

## Files

- `spam_detector.py`: main script that trains the model and allows interactive prediction.
- `sms_spam.csv`: sample dataset of spam/non-spam messages.
- `requirements.txt`: required Python packages.
- `app.py`: web application for browser-based spam checking.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the web app:
   ```bash
   python app.py
   ```
3. Open your browser to `http://127.0.0.1:5000` and type a message to check.

## Command-line option

If you want to use the terminal version instead, run:

```bash
python spam_detector.py
```

## Notes

- The script includes a custom `clean_text` step for basic text preprocessing.
- It evaluates the model on a test split and prints accuracy and classification metrics.
- This is a lightweight example meant for learning how text classification works.

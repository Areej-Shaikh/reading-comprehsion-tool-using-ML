import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

DATA_TRAIN = "data/processed/train.csv"
DATA_DEV   = "data/processed/dev.csv"
SAVE_DIR   = "models/model_b/traditional"

#clean text
def clean(text):
    return str(text).lower().strip()

#get distractor candidates based on cosine similarity
def get_distractor_candidates(article, correct_answer, vectorizer):
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
    if not sentences:
        return ["No distractor found", "No distractor found", "No distractor found"]


    correct_vec = vectorizer.transform([clean(correct_answer)])
    sentence_vecs = vectorizer.transform([clean(s) for s in sentences])


    sims = cosine_similarity(correct_vec, sentence_vecs)[0]

    sorted_indices = np.argsort(sims)

    mid = len(sorted_indices) // 2
    distractor_indices = sorted_indices[mid:mid+3]

    distractors = []
    for idx in distractor_indices:

        words = sentences[idx].split()[:15]
        distractors.append(' '.join(words))

#adding placeholders
    while len(distractors) < 3:
        distractors.append("Not enough content")

    return distractors[:3]

#get hints based on cosine similarity to question
def get_hints(article, question, vectorizer):
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
    if not sentences:
        return ["Read the passage carefully.",
                "Look for key words from the question.",
                "The answer is directly stated in the passage."]


    question_vec = vectorizer.transform([clean(question)])
    sentence_vecs = vectorizer.transform([clean(s) for s in sentences])


    sims = cosine_similarity(question_vec, sentence_vecs)[0]
    ranked = np.argsort(sims)[::-1]  

    hints = []
    for idx in ranked[:3]:
        words = sentences[idx].split()[:20]
        hints.append(' '.join(words))

    while len(hints) < 3:
        hints.append("Re-read the passage carefully.")

    hints = hints[::-1]
    return hints

#build dataset for training distractor ranker
def build_distractor_dataset(df):
    rows = []
    options = ["A", "B", "C", "D"]
    for _, row in df.iterrows():
        correct = row["answer"]
        for opt in options:
            text = clean(row["article"]) + " " + clean(row[opt])
            label = 0 if opt == correct else 1  # 1 = distractor, 0 = correct
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)
#train Model B
def train_model_b():
    print("Loading data...")
    train = pd.read_csv(DATA_TRAIN)
    dev   = pd.read_csv(DATA_DEV)

    print("Building distractor ranking dataset...")
    train_data = build_distractor_dataset(train)
    dev_data   = build_distractor_dataset(dev)

    X_train_text = train_data["text"]
    y_train      = train_data["label"]
    X_dev_text   = dev_data["text"]
    y_dev        = dev_data["label"]

    print("Vectorizing with One-Hot Encoding...")
    vectorizer = CountVectorizer(binary=True, max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev   = vectorizer.transform(X_dev_text)

    print("Training Distractor Ranker (Logistic Regression)...")
    model = LogisticRegression(max_iter=200, solver="saga", class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_dev)
    print("\n── Model B Results ──")
    print("Accuracy :", accuracy_score(y_dev, preds))
    print("Precision:", precision_score(y_dev, preds))
    print("Recall   :", recall_score(y_dev, preds))
    print("Macro F1 :", f1_score(y_dev, preds, average="macro"))
    print(classification_report(y_dev, preds))

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(SAVE_DIR, "distractor_ranker.pkl"))
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "distractor_vectorizer.pkl"))
    print(f"\nModel B saved to {SAVE_DIR}")
    print("All Model B artifacts saved successfully.")


if __name__ == "__main__":
    train_model_b()
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_TRAIN = "data/processed/train.csv"
DATA_DEV   = "data/processed/dev.csv"
SAVE_DIR   = "models/model_a/traditional"
OPTIONS    = ["A", "B", "C", "D"]

def make_verification_dataset(df):
    dfs = []
    for option in OPTIONS:
        temp = pd.DataFrame()
        temp["text"]  = (df["article"].fillna("") + " " + 
                         df["question"].fillna("") + " " + 
                         df[option].fillna(""))
        temp["label"] = (df["answer"] == option).astype(int)
        dfs.append(temp)
    return pd.concat(dfs, ignore_index=True)


def train_models():
    print("Loading data...")
    train = pd.read_csv(DATA_TRAIN)
    dev   = pd.read_csv(DATA_DEV)

    print("Building verification datasets...")
    train_verify = make_verification_dataset(train)
    dev_verify   = make_verification_dataset(dev)

    X_train_text = train_verify["text"]
    y_train      = train_verify["label"]
    X_dev_text   = dev_verify["text"]
    y_dev        = dev_verify["label"]

    print("Vectorizing with one hot encoding: ")
    vectorizer = CountVectorizer(binary=True, max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev   = vectorizer.transform(X_dev_text)
    models = {
    "Logistic Regression": LogisticRegression(
        max_iter=200, 
        solver="saga", 
        class_weight="balanced"  
    ),
    "SVM": LinearSVC(
        max_iter=200, 
        dual=False,
        class_weight="balanced"  
    ),
}
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "onehot_vectorizer.pkl"))
    print("Vectorizer saved.")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_dev)

        print(f"\n── {name} Results ──")
        print("Accuracy :", accuracy_score(y_dev, preds))
        print("Macro F1 :", f1_score(y_dev, preds, average="macro"))
        print(classification_report(y_dev, preds))

        filename = name.lower().replace(" ", "_") + "_verifier.pkl"
        joblib.dump(model, os.path.join(SAVE_DIR, filename))
        print(f"{name} saved to {SAVE_DIR}/{filename}")

    print("\nAll Model A artifacts saved successfully.")


if __name__ == "__main__":
    train_models()